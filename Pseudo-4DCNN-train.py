from __future__ import print_function

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
# from skimage.measure import compare_psnr as psnr

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

    usage = "Pseudo-4D CNN"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-D', '--path', type=str, default=None, dest='path',
        help='Loading 4D training and validation LF from this path: (default: %(default)s)')
    parser.add_argument(
        '-n', '--sample_num', type=int, default=25, dest='sample_num',
        help='Number of Samples in use: (default: %(default)s)')
    parser.add_argument(
        '-l', '--length', type=int, default=9, dest='length',
        help='Length of 3D LF: (default: %(default)s)')
    parser.add_argument(
        '-f', '--factor', type=int, default=2, dest='factor',
        help='Angular Upsampling factor: (default: %(default)s)')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=64, dest='batch_size',
        help='Batch Size: (default: %(default)s)')
    parser.add_argument(
        '-p', '--patch_size', type=int, default=48, dest='patch_size',
        help='Patch Size: (default: %(default)s)')
    parser.add_argument(
        '-s', '--prior_sensitive_loss', type=str2bool, default='False', dest='prior_sensitive_loss',
        help='Prior Sensitive Loss: (default: %(default)s)')

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

        print('Building Learnable Interpolation layer, init kernel\n',self.kern)

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
        print('Building 3d layer with shape ',filter_shape)
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
        prior_sensitive_loss = options['prior_sensitive_loss']
        print('Prior Sensitive Flag: ',prior_sensitive_loss)


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

        cn_interp_layers, cn_interp_params = self._init_interp_layer(upsample_factor,border_len,batch_size,x_res)

        layers = dict(rn_interp_layers,**cn_interp_layers)
        params = dict(rn_interp_params,**cn_interp_params)

        if model is not None:
            for k in params.iterkeys():
                params[k].set_value(model[k])

        proj = self._build_model(x, options, layers, params)

        weight_decay = theano.shared(numpy_floatX(0.), borrow=True)

        for v in params.itervalues():
            weight_decay += (v ** 2).sum()

        prior_weight_matrix = generate_prior_sensitive_weight(s_res,upsample_factor,border_len,prior_sensitive_loss)

        weight_m = theano.shared(prior_weight_matrix,borrow=False)

        cost =  tensor.mean(tensor.flatten((y - proj) ** 2 * weight_m.dimshuffle('x','x','x',0,1))) \
                + 1e-4 * weight_decay

        f_x = theano.function([x], proj, name='f_proj')

        return x, y, f_x, cost, params

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

        # rval_init = upsampled_inter_lf.dimshuffle(4,1,3,2,0)




        # rval = rval_init
        #
        # # print(rval.eval().shape)
        #
        # for i_n in range(len(options['conv_shape'])):
        #     rval = layers['rn_conv_'+str(i_n)].conv(rval)
        #     if i_n < len(options['conv_shape'])-1:
        #         rval = tensor.nnet.relu(rval) #+up_volume[:,0,:,:,:]
        #
        # inter_lf = (rval + rval_init).reshape((t_res,batch_size,s_up_res,x_res,y_res))
        #
        def _cstep(volume):
            return layers['cn_interp'].deconv(volume)

        upsampled_out_lf, _ = theano.scan(_cstep,sequences=[upsampled_inter_lf.dimshuffle(4,1,3,2,0)])

        # # for s_n in range(s_up_res):
        # #     init_volume = inter_lf[:,:,:,s_n,:]
        # #     up_volume = layers['rn_interp'].deconv(init_volume)
        # #     out_lf = tensor.set_subtensor(out_lf[:,:,:,s_n,:],up_volume)
        #
        # rval_init = upsampled_out_lf.dimshuffle(0,1,4,2,3).reshape((-1,1,t_up_res,x_res,y_res))
        #
        # rval = rval_init
        #
        # for i_n in range(len(options['conv_shape'])):
        #     rval = layers['cn_conv_'+str(i_n)].conv(rval)
        #     if i_n < len(options['conv_shape'])-1:
        #         rval = tensor.nnet.relu(rval)
        #
        # out = (rval+rval_init).reshape((s_up_res,batch_size,t_up_res,x_res,y_res)).dimshuffle(1,3,4,0,2)


        return  upsampled_out_lf.dimshuffle(1,2,3,0,4)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return range(len(minibatches)), minibatches


def load_model(path):
    npy = np.load(path)
    return npy.all()


def generate_prior_sensitive_weight(angular_size, upsample_factor, border_len, prior_strategy=False):

    up_angular_size = get_init_len(angular_size,upsample_factor,border_len)
    if prior_strategy:
        weight_matrix = np.zeros([up_angular_size,up_angular_size]).astype(config.floatX)
        for i in range(up_angular_size):
            for j in range(up_angular_size):
                if i % upsample_factor == 0 and j % upsample_factor == 0:
                    weight_matrix[i,j] = 0.1
                elif i % upsample_factor == 0 and not j % upsample_factor == 0:
                    weight_matrix[i,j] = 1.0
                elif not i % upsample_factor == 0 and j % upsample_factor == 0:
                    weight_matrix[i,j] = 1.0
                else:
                    weight_matrix[i,j] = 2.0
    else:
        weight_matrix = np.ones([up_angular_size,up_angular_size]).astype(config.floatX)

    print('Prior Sensitive Weight Matrix:\n',weight_matrix)

    return weight_matrix

def pred_error(f_pred,data,target):

    x = data
    y = target
    pred = f_pred(x)

    dmax = tensor.max(y)
    dmin = tensor.min(y)

    pred = np.round(pred * 255.0)
    y = np.round(y * 255.0)

    z = np.mean((y - pred) ** 2)
    #
    # z /= x.shape[0] * x.shape[3] * x.shape[4]
    rmse = np.sqrt(z)
    # print('RMSE: ',rmse.eval())
    psnr = 20 * np.log10(255.0 / rmse)
    # psnr = tensor.sum(psnr)

    return psnr

def load_data(path):
    """the data is scalaed in [0 1]"""

    f = h5py.File(path,'r')
    lr_data = np.asarray(f.get('train_data')[:], dtype=config.floatX)
    hr_data = np.asarray(f.get('train_label')[:], dtype=config.floatX)
    v_lr_data = np.asarray(f.get('valid_data')[:], dtype=config.floatX)
    v_hr_data = np.asarray(f.get('valid_label')[:], dtype=config.floatX)

    print('Reading LF data from ', path)
    print('Train data Size', lr_data.shape, ' Range: ',lr_data.max(),lr_data.min())
    print('Train label Size', hr_data.shape, ' Range: ',hr_data.max(),hr_data.min())
    print('Validation data Size', v_lr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())
    print('Validation label size', v_hr_data.shape, ' Range: ',v_lr_data.max(),v_lr_data.min())

    return lr_data, hr_data, v_lr_data, v_hr_data

def prefix_p(prefix, params):
    tp = OrderedDict()
    for kk, pp in params.items():
        tp['%s_%s' % (prefix, kk)] = params[kk]
    return tp


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def train_Pseudo_4DCNN(
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=1000,  # The maximum number of epoch to run
        dispFreq=1,  # Display to stdout the training progress every N updates
        lrate=1e-4,  # Learning rate for sgd (not used for adadelta and rmsprop)
        optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use,
        # not recommanded (probably need momentum and decaying learning rate).
        path = None,
        sample_num = 20,
        patch_size = 48,
        length = 7,
        factor = 3,
        validFreq=100,  # Compute the validation error after this number of update.
        saveFreq=200,  # Save the parameters after every saveFreq updates
        batch_size=64,  # The batch size during training and validateing.
        # Parameter for extra option
        momentum = 0,
        lmodel=True,  # Path to a saved model we want to start from.
        prior_sensitive_loss = False
):
    options = locals().copy()

    print('...Loading Data')
    train_path = './P4DCNN_Train_c1_s%d_l%d_f%d_p%d.hdf5' %(sample_num,length,factor,patch_size)
    saveto='P4DCNN_trial_f%d_v%d' %(factor,length)  # The best model will be saved there
    model_path='./model/P4DCNN_trial_f%d_v%d_backup.npy' %(factor,length) # The model path
    train_set_x, train_set_y, valid_set_x, valid_set_y = load_data(train_path)


    options['upsample_factor'] = factor
    options['interp_filter_shape'] = [1,1,5,5]

    border_len, lr_len, hr_len = up_scale_len(length,factor)
    options['input_shape'] = (batch_size,patch_size,patch_size,lr_len,lr_len)
    options['border_len'] = border_len

    options['conv_shape'] = [
        [64,1,3,9,9],
        [16,64,3,1,1],
        [1,16,3,5,5]
    ]

    options['prior_sensitive_loss'] = prior_sensitive_loss

    print("Model Options",options)
    print('...Building Model')

    net = Pseduo4DCNN(options)

    model = None
    if lmodel:
        model = load_model(model_path)

    (x, y, f_x, cost, params) = net.build_net(model)

    f_cost = theano.function([x, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(params.values()))
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, params, grads, x, y, cost)

    print('... Training')
    print("%d train examples" % ((train_set_x.shape[0] / batch_size) * batch_size))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = train_set_x.shape[0] // batch_size
    if saveFreq == -1:
        saveFreq = train_set_x.shape[0] // batch_size

    uidx = 0  # the number of update done

    try:
        for eidx in range(max_epochs):

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_set_x.shape[0], batch_size, shuffle=False)

            for train_index in kf[1]:
                uidx += 1
                # Select the random examples for this minibatch
                x = np.asarray([train_set_x[t, :, :, :, :] for t in train_index])
                y = np.asarray([train_set_y[t, :, :, :, :] for t in train_index])

                x = theano.shared(value=x, borrow=True).eval()
                y = theano.shared(value=y, borrow=True).eval()

                cost = f_grad_shared(x, y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch {:03d} Update {:03d} Cost {}'.format(eidx,uidx,cost))

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = params
                    p = dict()
                    for k in params.iterkeys():
                        p[k] = np.asarray(params[k].eval()).astype(config.floatX)
                    np.save('./model/'+saveto, p)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:

                    kv = get_minibatches_idx(valid_set_x.shape[0], batch_size, shuffle=False)

                    valid_psnr = []

                    for valid_index in kv[1]:
                        v_x = np.asarray([valid_set_x[v, :, :, :, :] for v in valid_index])
                        v_y = np.asarray([valid_set_y[v, :, :, :, :] for v in valid_index])

                        # v_x = theano.shared(value=v_x, borrow=True).eval()
                        # v_y = theano.shared(value=v_y, borrow=True).eval()

                        valid_psnr.append(pred_error(f_x,v_x,v_y))

                    history_errs.append([np.average(valid_psnr)] + [cost])

                    if (best_p is None or
                                cost <= np.array(history_errs)[:, -1].min()):

                        best_p = params
                        bad_counter = 0
                    print('Epoch {:03d} Update {:03d} Validation PNSR {:.2f}dB'.format(eidx,uidx,np.average(valid_psnr)))

                    if (len(history_errs) > patience and
                                cost >= np.array(history_errs)[:-patience, -1].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            lrate /= 10.0
                            print('Downing learning rate for ', lrate, '\n')
                            bad_counter = 0


    except KeyboardInterrupt:
        print("Training interupted\n")


if __name__ == '__main__':

    parser = opts_parser()
    args = parser.parse_args()

    path = args.path
    sample_num = args.sample_num
    length = args.length
    factor = args.factor
    patch_size = args.patch_size
    batch_size = args.batch_size
    prior_sensitive_loss = args.prior_sensitive_loss

    train_Pseudo_4DCNN(max_epochs=30,sample_num=sample_num,length=length,factor=factor,
                       patch_size=patch_size,batch_size=batch_size,path=path,
                       prior_sensitive_loss=prior_sensitive_loss)











