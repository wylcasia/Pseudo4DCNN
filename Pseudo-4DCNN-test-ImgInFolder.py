from __future__ import print_function
import os
import time
import math
from PIL import Image
import h5py
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet import conv3d
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs
from collections import OrderedDict
from argparse import ArgumentParser
import skimage.io as io
import scipy.io as sio
import skimage
from skimage.transform import resize
from skimage import color
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import gc
import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()


"""
a random number generator used to initialize weights
"""
SEED = 123
rng = np.random.RandomState(SEED)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def opts_parser():

    usage = "Pseudo-4D CNN Test"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='Loading 4D training and validation LF from this path: (default: %(default)s)')
    parser.add_argument(
        '-M', '--model', type=str, default=None, dest='model_path',
        help='Loading pre-trained model file from this path: (default: %(default)s)')
    parser.add_argument(
        '-S', '--save_path', type=str, default=None, dest='save_path',
        help='Save Upsampled LF to this path: (default: %(default)s)')
    parser.add_argument(
        '-e', '--ext', type=str, default='png', dest='ext',
        help='Format of view images: (default: %(default)s)')
    parser.add_argument(
        '--adjust_tone', type=float, default=0.0, dest='adjust_tone',
        help='Image Filename: (default: %(default)s)')
    parser.add_argument(
        '-f', '--factor', type=int, default=2, dest='factor',
        help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-t', '--train_length', type=int, default=7, dest='train_length',
        help='Training data length: (default: %(default)s)')
    parser.add_argument(
        '-c', '--crop_length', type=int, default=7, dest='crop_length',
        help='Crop Length from Initial LF: (default: %(default)s)')
    parser.add_argument(
        '-a', '--angular_size', type=int, default=14, dest='angular_size',
        help='Angular Resolution of Initial LF: (default: %(default)s)')
    parser.add_argument(
        '--save_results', type=str2bool, default=True, dest='save_results',
        help='Save Results or Not: (default: %(default)s)')


    return parser

def up_scale_len(length,factor):
    border = length % factor
    if border == 0:
        lr_len = length // factor
        hr_len = lr_len * factor
    elif border == 1:
        lr_len = length // factor + 1
        hr_len = (lr_len - 1) * factor + 1
    else:
        raise ValueError('Length {} factor {} border {} can not deal with.'.format(length,factor,border))
    return  border, lr_len, hr_len

def get_init_len(lr_len,factor,border):
    if border == 0:
        hr_len = lr_len * factor
    else:
        hr_len = (lr_len - 1)*factor + 1

    return hr_len

def AdjustTone(img,coef,norm_flag=False):

    log.info('--------------')
    log.info('Adjust Tone')

    tic = time.time()
    rgb = np.zeros(img.shape)
    img = np.clip(img,0.0,1.0)
    output = img ** (1/1.5)
    output = color.rgb2hsv(output)
    output[:,:,1] = output[:,:,1] * coef
    output = color.hsv2rgb(output)
    if norm_flag:
        r = output[:,:,0]
        g = output[:,:,1]
        b = output[:,:,2]
        rgb[:,:,0] = (r-r.min())/(r.max()-r.min())
        rgb[:,:,1] = (g-g.min())/(g.max()-g.min())
        rgb[:,:,2] = (b-b.min())/(b.max()-b.min())
    else:
        rgb = output

    log.info('IN Range: %.2f-%.2f' % (img.min(),img.max()))
    log.info('OUT Range: %.2f-%.2f' % (output.min(),output.max()))
    log.info("Elapsed time: %.2f sec" % (time.time() - tic))
    log.info('--------------')

    return  rgb

def FolderTo4DLF(path,ext,length):
    path_str = path+'/*.'+ext
    log.info('-'*40)
    log.info('Loading %s files from %s' % (ext, path) )
    img_data = io.ImageCollection(path_str)
    if len(img_data)==0:
        raise IOError('No .%s file in this folder' % ext)
    # print(len(img_data))
    # print img_data[3].shape
    N = int(math.sqrt(len(img_data)))
    if not(N**2==len(img_data)):
        raise ValueError('This folder does not have n^2 images!')

    [height,width,channel] = img_data[0].shape
    lf_shape = (N,N,height,width,channel)
    log.info('Initial LF shape: '+str(lf_shape))
    border = (N-length)/2
    if border<0:
        raise ValueError('Border {0} < 0'.format(border))
    out_lf_shape = (height, width, channel, length, length)
    log.info('Output LF shape: '+str(out_lf_shape))
    lf = np.zeros(out_lf_shape).astype(config.floatX)
    # save_path = './DATA/train/001/Coll/'
    for i in range(border,N-border,1):
        for j in range(border,N-border,1):
            indx = j + i*N
            im = color.rgb2ycbcr(np.uint8(img_data[indx]))
            lf[:,:,0, i-border,j-border] = im[:,:,0]/255.0
            lf[:,:,1:3,i-border,j-border] = im[:,:,1:3]
            # io.imsave(save_path+str(indx)+'.png',img_data[indx])
    log.info('LF Range:')
    log.info('Channel 1 [%.2f %.2f]' %(lf[:,:,0,:,:].max(),lf[:,:,0,:,:].min()))
    log.info('Channel 2 [%.2f %.2f]' %(lf[:,:,1,:,:].max(),lf[:,:,1,:,:].min()))
    log.info('Channel 3 [%.2f %.2f]' %(lf[:,:,2,:,:].max(),lf[:,:,2,:,:].min()))
    log.info('--------------------')
    return lf

def ImgTo4DLF(filename,unum,vnum,length,adjust_tone,save_sub_flag=False):

    if save_sub_flag:
        subaperture_path = os.path.splitext(filename)[0]+'_GT/'
        if not(os.path.exists(subaperture_path)):
            os.mkdir(subaperture_path)

    rgb_uint8 = io.imread(filename)
    rgb = np.asarray(skimage.img_as_float(rgb_uint8))
    log.info('Image Shape: %s' % str(rgb.shape))

    height = rgb.shape[0]/vnum
    width = rgb.shape[1]/unum
    channel = rgb.shape[2]

    if channel > 3:
        log.info('  Bands/Channels >3 Convert to RGB')
        rgb = rgb[:,:,0:3]
        channel = 3

    if adjust_tone > 0.0:
        rgb = AdjustTone(rgb,adjust_tone)

    lf_shape = (height, width, channel, vnum, unum)
    lf = np.zeros(lf_shape).astype(config.floatX)
    log.info('Initial LF shape: '+str(lf_shape))
    for i in range(vnum):
        for j in range(unum):
            im = rgb[i::vnum,j::unum,:]
            if save_sub_flag:
                subaperture_name = subaperture_path+'View_%d_%d.png' %(i+1,j+1)
                io.imsave(subaperture_name,im)
            lf[:,:,:,i,j] = color.rgb2ycbcr(im)
            lf[:,:,0,i,j] = lf[:,:,0,i,j]/255.0

    if unum % 2 == 0:
        border = (unum-length)/2 + 1
        u_start_indx = border
        u_stop_indx = unum - border + 1
        v_start_indx = border
        v_stop_indx = vnum - border + 1
    else:
        border = (unum-length)/2
        u_start_indx = border
        u_stop_indx = unum - border
        v_start_indx = border
        v_stop_indx = vnum - border

    if border<0:
        raise ValueError('Border {0} < 0'.format(border))

    out_lf = lf[:,:,:,v_start_indx:v_stop_indx,u_start_indx:u_stop_indx]
    log.info('Output LF shape: '+str(out_lf.shape))

    log.info('LF Range:')
    log.info('Channel 1 [%.2f %.2f]' %(out_lf[:,:,0,:,:].max(),out_lf[:,:,0,:,:].min()))
    log.info('Channel 2 [%.2f %.2f]' %(out_lf[:,:,1,:,:].max(),out_lf[:,:,1,:,:].min()))
    log.info('Channel 3 [%.2f %.2f]' %(out_lf[:,:,2,:,:].max(),out_lf[:,:,2,:,:].min()))
    log.info('--------------------')
    return out_lf

def del_files(path,ext):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                os.remove(os.path.join(root, name))


def bilinear_kernel_1D(ratio=3, normalize=True):
    half_kern = tensor.arange(1, ratio + 1, dtype=theano.config.floatX)
    kern = tensor.concatenate([half_kern, half_kern[-2::-1]])
    if normalize:
        kern /= ratio
    return kern

def bilinear_kernel_2D(ratio=3, normalize=True):
    hkern = bilinear_kernel_1D(ratio=ratio, normalize=normalize).dimshuffle('x', 0)
    vkern = bilinear_kernel_1D(ratio=ratio, normalize=normalize).dimshuffle(0, 'x')
    kern = hkern * vkern
    return kern


class DeconvLayer(object):
    def __init__(self,ratio=3,border=0,batch_size=None,num_input_channels=None):
        ##filter_shape is [num_out_channels,num_in_channels,width,height]
        # self.in1, self.in2 = upsampled_shape
        # self.upscale_factor = upscale_factor
        # kernel_size = filter_shape[2]
        # ### Centre location of the filter for which value is calculated
        # if kernel_size % 2 == 1:
        #     centre_location = numpy_floatX(upscale_factor - 1)
        # else:
        #     centre_location = numpy_floatX(upscale_factor - 0.5)
        #
        # bilinear = np.zeros([filter_shape[2], filter_shape[3]])
        # for x in range(filter_shape[2]):
        #     for y in range(filter_shape[3]):
        #         ##Interpolation Calculation
        #         value = (1 - abs((x - centre_location)/ numpy_floatX(upscale_factor))) \
        #                 * (1 - abs((y - centre_location)/ numpy_floatX(upscale_factor)))
        #         bilinear[x, y] = value
        # weights = np.zeros(filter_shape)
        # for i in range(filter_shape[0]):
        #     weights[i, i, :, :] = bilinear
        #
        # print('Initial Interpolation Weights\n', weights)

        self.ratio = ratio
        self.border = border
        self.kern = bilinear_kernel_2D(ratio=self.ratio, normalize=True).eval()

        self.W = theano.shared(self.kern, borrow=True)

        self.batch_size = batch_size
        self.num_in_channels = num_input_channels

        self.params = [self.W, ]

        # print('Building Learnable Interpolation layer, init kernel\n',self.kern)

    def deconv(self,input):
        try:
            up_bs = self.batch_size * self.num_in_channels
        except TypeError:
            up_bs = None

        row, col = input.shape[2:]

        up_input = input.reshape((-1, 1, row, col))

        # # concatenating the first and last row and column
        # # first and last row
        # concat_mat = tensor.concatenate((up_input[:, :, :1, :], up_input,
        #                             up_input[:, :, -1:, :]), axis=2)
        # # first and last col
        # concat_mat = tensor.concatenate((concat_mat[:, :, :, :1], concat_mat,
        #                             concat_mat[:, :, :, -1:]), axis=3)
        # concat_col = col + 2

        # pad_col = 2 * self.ratio - (self.ratio - 1) // 2 - 1
        # pad_row = 3

        up_col = get_init_len(col,self.ratio,self.border)

        upsampled_mat = conv2d_grad_wrt_inputs(output_grad=up_input,
                                               filters=self.W[np.newaxis, np.newaxis, :, :],
                                               input_shape=(up_bs, 1, row, up_col),
                                               filter_shape=(1, 1, None, None),
                                               border_mode='half',
                                               subsample=(1, self.ratio),
                                               filter_flip=True,
                                               filter_dilation=(1, 1))
        return upsampled_mat.reshape((input.shape[0], input.shape[1], row, up_col))

class Conv2DLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape):
        """
        Allocate a c with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type name: str
        :param name: given a special name for the ConvPoolLayer
        """

        # self.filter_shape = filter_shape
        # self.image_shape = image_shape
        # self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
        #            np.prod(poolsize))
        # initialize weights with random weights

        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.normal(0, 1e-3, size=filter_shape),
                dtype=config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(np.zeros(filter_shape[0],).astype(config.floatX), borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def conv(self, input):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            border_mode='half'
        )
        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        return output

class Conv3DLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape):
        """
        Allocate a c with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type name: str
        :param name: given a special name for the ConvPoolLayer
        """

        # self.filter_shape = filter_shape
        # self.image_shape = image_shape
        # self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
        #            np.prod(poolsize))
        # initialize weights with random weights

        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        # print('Building 3d layer with shape ',filter_shape)
        self.W = theano.shared(
            np.asarray(
                rng.normal(0, 1e-3, size=filter_shape),
                dtype=config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(np.zeros(filter_shape[0]).astype(config.floatX), borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def conv(self, input):
        # convolve input feature maps with filters
        conv_out = conv3d(
            input=input,
            filters=self.W,
            border_mode='half'
        )
        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x','x')

        return output

class Pseduo4DCNN(object):

    def __init__(self,options):
        self.options = options

    def build_net(self, model):
        options = self.options

        input_shape = options['input_shape']
        batch_size = input_shape[0]
        x_res = input_shape[1]
        y_res = input_shape[2]
        s_res = input_shape[3]
        t_res = input_shape[4]
        border_len = options['border_len']


        upsample_factor = options['upsample_factor']

        # self.rn_interp_filter_shape = options['interp_filter_shape']
        # self.rn_interp_shape = [batch_size,1,x_res,s_res]
        # self.rn_upsampled_shape = [x_res,(s_res-1)*self.upsample_factor+1]
        #
        # self.cn_interp_filter_shape = options['interp_filter_shape']
        # self.cn_interp_shape = [batch_size,1,y_res,t_res]
        # self.cn_upsampled_shape = [y_res,(t_res-1)*self.upsample_factor+1]

        conv_shape = options['conv_shape']

        x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='x')
        y = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='y')

        rn_interp_layers, rn_interp_params = self._init_interp_layer(upsample_factor,border_len,batch_size,y_res)

        rn_conv_layers, rn_conv_params = self._init_conv_layer(conv_shape)

        cn_interp_layers, cn_interp_params = self._init_interp_layer(upsample_factor,border_len,batch_size,x_res)

        cn_conv_layers, cn_conv_params = self._init_conv_layer(conv_shape)


        rn_layers = dict(prefix_p('rn',rn_interp_layers), **(prefix_p('rn',rn_conv_layers)))
        rn_params = dict(prefix_p('rn',rn_interp_params), **(prefix_p('rn',rn_conv_params)))
        # rn_params = prefix_p('rn',rn_conv_params)

        cn_layers = dict(prefix_p('cn',cn_interp_layers), **(prefix_p('cn',cn_conv_layers)))
        cn_params = dict(prefix_p('cn',cn_interp_params), **(prefix_p('cn',cn_conv_params)))
        # cn_params = prefix_p('cn',cn_conv_params)

        layers = dict(rn_layers,**cn_layers)
        params = dict(rn_params,**cn_params)

        if model is not None:
            for k in params.iterkeys():
                params[k].set_value(model[k])

        proj = self._build_model(x, options, layers, params)

        f_x = theano.function([x], proj, name='f_proj')

        return x, y, f_x

    def _init_interp_layer(self, ratio, border, batch_size, num_input_channels):

        layers = OrderedDict()
        params = OrderedDict()

        layers['interp'] = DeconvLayer(ratio, border, batch_size, num_input_channels)
        params['interp'] = layers['interp'].params[0]

        return layers, params

    def _init_conv_layer(self, conv_shape):

        layers = OrderedDict()
        params = OrderedDict()

        for i in range(len(conv_shape)):
            layers['conv_'+str(i)] = Conv3DLayer(conv_shape[i])
            params['conv_'+str(i)+'_w'] = layers['conv_'+str(i)].params[0]
            params['conv_'+str(i)+'_b'] = layers['conv_'+str(i)].params[1]

        return layers, params

    def _build_model(self, input_lf, options, layers, params):

        # assert input_lf.eval().shape == options['input_shape']

        input_shape = options['input_shape']

        upsample_factor = options['upsample_factor']

        border_len = options['border_len']

        batch_size = input_shape[0]
        x_res = input_shape[1]
        y_res = input_shape[2]
        s_res = input_shape[3]
        t_res = input_shape[4]

        t_up_res = get_init_len(t_res,upsample_factor,border_len)
        s_up_res = get_init_len(s_res,upsample_factor,border_len)

        # inter_lf_shape = [batch_size,x_res,y_res,s_up_res,t_res]
        # out_lf_shape = [batch_size,x_res,y_res,s_up_res,t_up_res]

        # inter_lf = theano.shared(np.zeros(inter_lf_shape).astype(config.floatX), borrow=True)
        # out_lf = theano.shared(np.zeros(out_lf_shape).astype(config.floatX), borrow=True)

        def _rstep(volume):
            return layers['rn_interp'].deconv(volume)

        upsampled_inter_lf, _  = theano.scan(_rstep, sequences=[input_lf.dimshuffle(4,0,2,1,3)])

        # print(rval.eval().shape)


        # for t_n in range(t_res):
        #     init_volume = input_lf[:,:,:,:,t_n]
        #     up_volume = layers['rn_interp'].deconv(init_volume.dimshuffle(0,2,1,3))
        #     inter_lf = tensor.set_subtensor(inter_lf[:,:,:,:,t_n],up_volume.dimshuffle(0,2,1,3))

        rval_init = upsampled_inter_lf.dimshuffle(0,1,4,3,2).reshape((-1,1,s_up_res,x_res,y_res))

        rval = rval_init

        # print(rval.eval().shape)

        for i_n in range(len(options['conv_shape'])):
            rval = layers['rn_conv_'+str(i_n)].conv(rval)
            if i_n < len(options['conv_shape'])-1:
                rval = tensor.nnet.relu(rval) #+up_volume[:,0,:,:,:]

        inter_lf = (rval + rval_init).reshape((t_res,batch_size,s_up_res,x_res,y_res))

        def _cstep(volume):
            return layers['cn_interp'].deconv(volume)

        upsampled_out_lf, _ = theano.scan(_cstep,sequences=[inter_lf.dimshuffle(2,1,3,4,0)])

        # for s_n in range(s_up_res):
        #     init_volume = inter_lf[:,:,:,s_n,:]
        #     up_volume = layers['rn_interp'].deconv(init_volume)
        #     out_lf = tensor.set_subtensor(out_lf[:,:,:,s_n,:],up_volume)

        rval_init = upsampled_out_lf.dimshuffle(0,1,4,2,3).reshape((-1,1,t_up_res,x_res,y_res))

        rval = rval_init

        for i_n in range(len(options['conv_shape'])):
            rval = layers['cn_conv_'+str(i_n)].conv(rval)
            if i_n < len(options['conv_shape'])-1:
                rval = tensor.nnet.relu(rval)

        out = (rval+rval_init).reshape((s_up_res,batch_size,t_up_res,x_res,y_res)).dimshuffle(1,3,4,0,2)


        return  out

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def load_model(path):
    npy = np.load(path)
    return npy.all()


def load_data(path):
    """the data is scalaed in [0 1]"""

    f = h5py.File(path,'r')
    lr_data = np.asarray(f.get('train_data')[:], dtype=config.floatX)
    hr_data = np.asarray(f.get('train_label')[:], dtype=config.floatX)
    v_lr_data = np.asarray(f.get('valid_data')[:], dtype=config.floatX)
    v_hr_data = np.asarray(f.get('valid_label')[:], dtype=config.floatX)

    log.info('Reading LF data from ', path)
    log.info('Train data Size', lr_data.shape, ' Range: ',lr_data.max(),lr_data.min())
    log.info('Train label Size', hr_data.shape, ' Range: ',hr_data.max(),hr_data.min())
    log.info('Validation data Size', v_lr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())
    log.info('Validation label size', v_hr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())

    return lr_data, hr_data, v_lr_data, v_hr_data

def prefix_p(prefix, params):
    tp = OrderedDict()
    for kk, pp in params.items():
        tp['%s_%s' % (prefix, kk)] = params[kk]
    return tp

def getSceneNameFromPath(path,ext):
    sceneNamelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext):
                sceneName = os.path.splitext(name)[0]
                sceneNamelist.append(sceneName)

    sceneNamelist.sort()

    return tuple(sceneNamelist)

def test_Pseudo_4DCNN(
        path = None,
        model_path = None,
        save_path = None,
        ext = 'png',
        img_filename = None,
        train_length = 7,
        angular_size = 14,
        crop_length = 7,
        factor = 3,
        adjust_tone = 0.0,
        save_results = False
):
    options = locals().copy()

    if path is not None:
        log.info('='*40)
        if not os.path.exists(path):
            raise IOError('No such folder: {}'.format(path))
        if save_path is None:
            save_path = path+'_eval/'
        if not os.path.exists(save_path):
            log.warning('No such path for saving Our results, creating dir {}'
                        .format(save_path))
            os.mkdir(save_path)

        sceneNameTuple = getSceneNameFromPath(path,ext)

        if len(sceneNameTuple) == 0:
            raise IOError('Not any .%s file found in %s' %(ext,path))
        else:
            scene_num = len(sceneNameTuple)
    else:
        raise NameError('No folder given.')

    log_file = os.path.join(save_path,'test_%s.log'
                            % datetime.datetime.now().strftime("%Y%m%d%H%M"))
    fh = logging.FileHandler(log_file)
    log.addHandler(fh)

    total_PSNR = []
    total_SSIM = []
    total_Elapsedtime = []

    performacne_index_file = os.path.join(save_path,'performance_stat.mat')

    options['upsample_factor'] = factor
    options['path'] = path
    options['model_path'] = model_path
    options['save_path'] = save_path
    options['ext'] = ext
    options['train_length'] = train_length
    options['angular_size'] = angular_size
    options['crop_length'] = crop_length
    options['save_results'] = save_results

    for scene in sceneNameTuple:
        log.info('='*20+str(scene)+'='*20)
        if save_results:
            our_save_path = save_path + scene + '_OURS'
            GT_save_path = save_path + scene + '_GT'
            if os.path.isdir(our_save_path):
                log.info('-'*20)
                del_files(our_save_path,'png')
                log.warning('Ours Save Path %s exists, delete all .png files' % our_save_path)
            else:
                os.mkdir(our_save_path)

            if os.path.isdir(GT_save_path):
                del_files(GT_save_path,'png')
                log.info('GT path %s exists, delete all .png files' % GT_save_path)
            else:
                os.mkdir(GT_save_path)

        tic = time.time()
        img_filename = os.path.join(path,scene+'.'+ext)

        lf = ImgTo4DLF(filename=img_filename,vnum=angular_size,unum=angular_size,length=crop_length,
                       adjust_tone=adjust_tone,save_sub_flag=False)

        input_lf = lf[:,:,0,0:crop_length:factor,0:crop_length:factor]

        log.info("Elapsed time: %.2f sec" % (time.time() - tic))

        border_len, lr_len, hr_len = up_scale_len(crop_length,factor)

        x_res = lf.shape[0]
        y_res = lf.shape[1]
        channel = lf.shape[2]
        s_res = lf.shape[3]
        t_res = lf.shape[4]

        options['input_shape'] = (1,x_res,y_res,lr_len,lr_len)
        options['border_len'] = border_len
        options['conv_shape'] = [
            [32,1,3,9,9],
            [16,32,3,1,1],
            [1,16,3,5,5]
        ]

        log.info('-'*40)
        log.info("Model Options\n"+str(options))
        log.info('-'*40)
        log.info('...Building Model')

        net = Pseduo4DCNN(options)

        model_file = 'P4DCNN_trial_f%d_v%d.npy' %(factor,train_length)

        if not os.path.exists(os.path.join(model_path,model_file)):
            raise IOError('No Such Model File %s', os.path.join(model_path,model_file))
        else:
            log.info('Loading pre-trained model from %s' % os.path.join(model_path,model_file))
            model = load_model(os.path.join(model_path,model_file))

        s_time = time.time()
        (x, y, f_x) = net.build_net(model)
        out_lf = f_x(input_lf[np.newaxis,:,:,:,:])[0]
        process_time = time.time() - s_time

        PSNR = []
        SSIM = []

        log.info('-'*40)
        log.info('Evaluation......')

        for s_n in xrange(hr_len):
            for t_n in xrange(hr_len):
                gt_img = lf[:,:,0,s_n,t_n]
                # print('GT range: %.2f-%.2f' %(gt_img.min(),gt_img.max()))
                view_img = np.clip(out_lf[:,:,s_n,t_n],gt_img.min(),gt_img.max())
                # if not(u % factor == 0 and v % factor == 0):
                # this_test_loss = np.sqrt(mse(view_img,gt_img))
                # this_PSNR = 20*math.log(1.0/this_test_loss,10)
                this_PSNR = psnr(view_img,gt_img)
                this_SSIM = ssim(view_img,gt_img)
                log.info('View %.2d_%.2d: PSNR: %.2fdB SSIM: %.4f' %(s_n+1, t_n+1, this_PSNR, this_SSIM))
                PSNR.append(this_PSNR)
                SSIM.append(this_SSIM)

                if save_results:
                    filename = os.path.join(our_save_path,'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    GTname = os.path.join(GT_save_path,'View_'+str(s_n+1)+'_'+str(t_n+1)+'.png')
                    out_img = np.zeros((x_res,y_res,3))
                    gt_out_img = np.zeros((x_res,y_res,3))

                    out_img[:,:,0] = np.clip(view_img*255.0,16.0,235.0)
                    gt_out_img[:,:,0] = np.clip(gt_img*255.0,16.0,235.0)
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,0].max(),out_img[:,:,0].min()))
                    out_img[:,:,1:3] = lf[:,:,1:3,s_n,t_n]
                    gt_out_img[:,:,1:3] = lf[:,:,1:3,s_n,t_n]
                    # print('Max: %.2f Min: %.2f' %(out_img[:,:,1].max(),out_img[:,:,1].min()))

                    out_img = color.ycbcr2rgb(out_img)
                    out_img = np.clip(out_img,0.0,1.0)
                    out_img = np.uint8(out_img*255.0)

                    gt_out_img = color.ycbcr2rgb(gt_out_img)
                    gt_out_img = np.clip(gt_out_img,0.0,1.0)
                    gt_out_img = np.uint8(gt_out_img*255.0)

                    io.imsave(filename,out_img)
                    io.imsave(GTname,gt_out_img)

        log.info('='*40)
        total_PSNR.append(np.mean(np.array(PSNR)))
        total_SSIM.append(np.mean(np.array(SSIM)))
        total_Elapsedtime.append(process_time)
        log.info("Average PSNR: %.2f dB\nSSIM: %.4f\nElapsed time: %.2f sec" % (np.mean(np.array(PSNR)),
                                                                             np.mean(np.array(SSIM)), process_time))
        gc.collect()
        log.info('='*40)


    log.info('='*6+'Average Performance on %d scenes' % scene_num+'='*6)
    log.info('PSNR: %.2f dB' % np.mean(np.array(total_PSNR)))
    log.info('SSIM: %.4f' % np.mean(np.array(total_SSIM)))
    log.info('Elapsed Time: %.2f sec' % np.mean(np.array(total_Elapsedtime)))
    log.info('='*40)

    embeded = dict(NAME=sceneNameTuple,PSNR=np.array(total_PSNR),SSIM=np.array(total_SSIM),TIME=np.array(total_Elapsedtime))
    sio.savemat(performacne_index_file,embeded)


if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    model_path = args.model_path
    save_path = args.save_path
    ext = args.ext
    factor = args.factor
    train_length = args.train_length
    crop_length = args.crop_length
    adjust_tone = args.adjust_tone
    save_results = args.save_results
    angular_size = args.angular_size

    test_Pseudo_4DCNN(path=path,model_path=model_path,ext=ext,factor=factor, adjust_tone=adjust_tone,
                      train_length=train_length,crop_length=crop_length,
                      angular_size=angular_size,save_results=save_results)











