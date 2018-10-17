# -*- coding: UTF-8 –*-
import time
import string
from PIL import Image, ImageChops
from PIL.GifImagePlugin import getheader, getdata
import os
from argparse import ArgumentParser
import imageio
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()


__author__ = 'Yunlong Wang'

def opts_parser():

    usage = "Make GIF from LF folder containing sub-aperture images with name like 'View_X_X' "
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='LF folder: (default: %(default)s)')
    parser.add_argument(
        '-S', '--save_name', type=str, default='lf_save.gif', dest='save_name',
        help='Save Upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-E', '--ext', type=str, default='png', dest='ext',
        help='Format of view images: (default: %(default)s)')
    parser.add_argument(
        '-L', '--length', type=int, default=7, dest='length',
        help='Angular : (default: %(default)s)')
    parser.add_argument(
        '-T', '--duration', type=float, default=0.05, dest='duration',
        help='Time Duration : (default: %(default)s)')

    return parser

def intToBin(i):
    """ 把整型数转换为双字节 """
    # 先分成两部分,高8位和低8位
    i1 = i % 256
    i2 = int( i/256)
    # 合成小端对齐的字符串
    return chr(i1) + chr(i2)
def getheaderAnim(im):
    """ 生成动画文件头 """
    bb = "GIF89a"
    bb += intToBin(im.size[0])
    bb += intToBin(im.size[1])
    bb += "\x87\x00\x00"  #使用全局颜色表
    return bb
def getAppExt(loops=0):
    """ 应用扩展,默认为0,为0是表示动画是永不停止
    """
    bb = "\x21\xFF\x0B"  # application extension
    bb += "NETSCAPE2.0"
    bb += "\x03\x01"
    if loops == 0:
        loops = 2**16-1
    bb += intToBin(loops)
    bb += '\x00'  # end
    return bb

def getGraphicsControlExt(duration=0.1):
    """ 设置动画时间间隔 """
    bb = '\x21\xF9\x04'
    bb += '\x08'  # no transparancy
    bb += intToBin( int(duration*100) ) # in 100th of seconds
    bb += '\x00'  # no transparant color
    bb += '\x00'  # end
    return bb

def _writeGifToFile(fp, images, durations, loops):
    """ 把一系列图像转换为字节并存入文件流中
    """
    # 初始化
    frames = 0
    previous = None
    for im in images:
        if not previous:
            # 第一个图像
            # 获取相关数据
            palette = getheader(im)[0][3]  #取第一个图像的调色板
            # palette = Image.ADAPTIVE
            data = getdata(im)
            imdes, data = data[0], data[1:]
            header = getheaderAnim(im)
            appext = getAppExt(loops)
            graphext = getGraphicsControlExt(durations[0])

            # 写入全局头
            fp.write(header)
            fp.write(palette)
            fp.write(appext)

            # 写入图像
            fp.write(graphext)
            fp.write(imdes)
            for d in data:
                fp.write(d)

        else:
            # 获取相关数据
            data = getdata(im)
            imdes, data = data[0], data[1:]
            graphext = getGraphicsControlExt(durations[frames])

            # 写入图像
            fp.write(graphext)
            fp.write(imdes)
            for d in data:
                fp.write(d)
        # 准备下一个回合
        previous = im.copy()
        frames = frames + 1

    fp.write(";")  # 写入完成
    return frames

def writeGif(filename, images, duration=0.1, loops=0, dither=1):
    """ writeGif(filename, images, duration=0.1, loops=0, dither=1)
    从输入的图像序列中创建GIF动画
    images 是一个PIL Image [] 或者 Numpy Array
    """
    images2 = []
    # 先把图像转换为PIL格式
    for im in images:

        if isinstance(im,Image.Image): #如果是PIL Image
            images2.append( im.convert('P',dither=dither,palette=Image.ADAPTIVE) )

        elif np and isinstance(im, np.ndarray): #如果是Numpy格式
            if im.dtype == np.uint8:
                pass
            elif im.dtype in [np.float32, np.float64]:
                im = (im*255).astype(np.uint8)
            else:
                im = im.astype(np.uint8)
            # 转换
            if len(im.shape)==3 and im.shape[2]==3:
                im = Image.fromarray(im,'RGB').convert('P',dither=dither,palette=Image.ADAPTIVE)
            elif len(im.shape)==2:
                im = Image.fromarray(im,'L').convert('P',dither=dither,palette=Image.ADAPTIVE)
            else:
                raise ValueError("图像格式不正确")
            images2.append(im)

        else:
            raise ValueError("未知图像格式")

    # 检查动画播放时间
    durations = [duration for im in images2]
    # 打开文件
    fp = open(filename, 'wb')
    # 写入GIF
    try:
        n = _writeGifToFile(fp, images2, durations, loops)
    finally:
        fp.close()
    return n

############################################################
## 将多帧位图合成为一幅gif图像
def images2gif( images, giffile, durations=0.05, loops = 1):
    seq = []
    for i in range(len(images)):
        # print(images[i])
        im = Image.open(images[i])
        # background = Image.new('RGB', im.size, (255,255,255))
        # background.paste(im, (0,0))
        # seq.append(background)
        seq.append(im)
    frames = writeGif( giffile, seq, durations, loops)
    print(frames, 'images has been merged to', giffile)

def getNameTupleInOrder(path,length,prefix,ext):
    images = []
    # First Loop
    for n in range(length**2):
        row_n = (n+1) // length
        col_n = (n+1) % length
        if col_n == 0:
            row = row_n
            if row_n % 2 == 0:
                col = 1
            else:
                col = length
        else:
            row = row_n + 1
            if row_n % 2 == 0:
                col = col_n
            else:
                col = length - col_n + 1
        images.append(os.path.join(path,prefix+'_{r}_{c}.'.format(r=row,c=col)+ext))
    # Second Loop
    for n in range(length**2):
        col_n = (n+1) // length
        row_n = (n+1) % length
        if row_n == 0:
            col = length - col_n + 1
            if col_n % 2 == 0:
                row = length
            else:
                row = 1
        else:
            col = length - col_n
            if col_n % 2 == 0:
                row = length - row_n + 1
            else:
                row = row_n
        images.append(os.path.join(path,prefix+'_{r}_{c}.'.format(r=row,c=col)+ext))

    return tuple(images)



if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    save_name = args.save_name
    length = args.length
    ext = args.ext
    duration = args.duration

    print('='*15+'Summary'+'='*15)
    print('LF path: %s' % path)
    print('Length: %d' % length)
    print('Ext: %s' % ext)
    print('Duration: %f' % duration)
    print('Save Name: %s' % save_name)
    print('='*35)

    images = getNameTupleInOrder(path,length,'View',ext)

    # im = Image.open(images[0])
    #
    # im.save(os.path.join(path,save_name),save_all=True,append_images=[Image.open(filename) for filename in images],
    #         loop=5,duration=200)
    #
    # print('images has been merged to '+save_name)


    # images2gif(images,os.path.join(path,save_name), durations = duration)
    files = []
    gifName = os.path.join(path,save_name)
    for image in images:
        files.append(imageio.imread(image))
    kargs = { 'duration': duration }
    imageio.mimsave(gifName, files, 'GIF', **kargs)
    print('%d Images merged into %s' %(2*length**2,gifName))










