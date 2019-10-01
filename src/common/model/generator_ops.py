import tensorflow as tf
from common.model import ops
from common.model.ops import up_sample, upsample_ps, leaky_relu


def _block(x, labels, out_channels, num_classes, name, conv=ops.snconv2d, kernel=(3, 3), strides=(1, 1),
           dilations=(1, 1), act=leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in the generator.

    Compared with block, it takes into account that there are different classes.

    Args:
      x: The 4D input vector.
      labels: The conditional labels in the generation.
      out_channels: Number of features in the output layer.
      num_classes: Number of classes in the labels.
      name: Scope name
      conv: Convolution function. Options conv2d or snconv2d
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      act: The activation function used in the block. (Default value = leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. (Default value = 'avg')
      padding:  Type of padding  (Default value = 'SAME')
      conditional: A flag that determines if conditional batch norm should be used.
      If false, standard batch norm is used. (Default value = False)

    Returns:
      A tensor representing the output of the operation.

    """
    with tf.variable_scope(name):
        x_0 = x
        if num_classes is not None:
            bn0 = ops.ConditionalBatchNorm(num_classes, name='cbn_0')
            bn1 = ops.ConditionalBatchNorm(num_classes, name='cbn_1')
            x = bn0(x, labels)
        else:
            bn0 = ops.BatchNorm(name='bn_0') #TODO
            bn1 = ops.BatchNorm(name='bn_1')
            x = x
        x = act(x)
        x = up_sampling(x, pooling, out_channels, conv, kernel, strides, 'conv1', padding)
        if num_classes is not None:
            x = bn1(x, labels)
        else:
            x = x
        x = act(x)
        x = conv(x, out_channels, kernel, dilations=dilations, name='conv2', padding=padding)
        x_0 = up_sampling(x_0, pooling, out_channels, conv, kernel, strides, 'conv3', padding)
        return x_0 + x


def sn_block(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1),
             act=leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in the generator. It uses 2D conv with spectral normalization
    Args:
      x: The 4D input vector.
      labels: The conditional labels in the generation.
      out_channels: Number of features in the output layer.
      num_classes: Number of classes in the labels.
      name: Scope name
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      act: The activation function used in the block. (Default value = leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. (Default value = 'avg')
      padding:  Type of padding  (Default value = 'SAME')

    Returns:
      A tensor representing the output of the operation.

    """
    return _block(x, None, out_channels, None, "sn_" + name, ops.snconv2d, kernel, strides, dilations, act, pooling,
                  padding)


def sn_block_conditional(x, labels, out_channels, num_classes, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1),
                         act=leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in the generator.

    Compared with block, it is optimised to work with conditional GAN. It uses 2D conv with spectral normalization

    Args:
      x: The 4D input vector.
      labels: The conditional labels in the generation.
      out_channels: Number of features in the output layer.
      num_classes: Number of classes in the labels.
      name: Scope name
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      act: The activation function used in the block. (Default value = leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. (Default value = 'avg')
      padding:  Type of padding  (Default value = 'SAME')

    Returns:
      A tensor representing the output of the operation.

    """
    return _block(x, labels, out_channels, num_classes, "sn_c_" + name, ops.snconv2d, kernel, strides, dilations, act,
                  pooling, padding)


def block(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1),
          act=leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in the generator. It uses standard 2D conv
    Args:
      x: The 4D input vector.
      labels: The conditional labels in the generation.
      out_channels: Number of features in the output layer.
      num_classes: Number of classes in the labels.
      name: Scope name
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      act: The activation function used in the block. (Default value = leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. (Default value = 'avg')
      padding:  Type of padding  (Default value = 'SAME')

    Returns:
      A tensor representing the output of the operation.

    """
    return _block(x, None, out_channels, None, name, ops.conv2d, kernel, strides, dilations, act, pooling,
                  padding)


def scaled_block(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1),
          act=leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in the generator. It uses standard scaled 2D conv
    Args:
      x: The 4D input vector.
      labels: The conditional labels in the generation.
      out_channels: Number of features in the output layer.
      num_classes: Number of classes in the labels.
      name: Scope name
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      act: The activation function used in the block. (Default value = leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. (Default value = 'avg')
      padding:  Type of padding  (Default value = 'SAME')

    Returns:
      A tensor representing the output of the operation.

    """
    return _block(x, None, out_channels, None, name, ops.scaled_conv2d, kernel, strides, dilations, act, pooling,
                  padding)


def block_conditional(x, labels, out_channels, num_classes, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1),
                      act=leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in the generator.

    Compared with block, it is optimised to work with conditional GAN. It uses standard 2D conv

    Args:
      x: The 4D input vector.
      labels: The conditional labels in the generation.
      out_channels: Number of features in the output layer.
      num_classes: Number of classes in the labels.
      name: Scope name
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      act: The activation function used in the block. (Default value = leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. (Default value = 'avg')
      padding:  Type of padding  (Default value = 'SAME')

    Returns:
      A tensor representing the output of the operation.

    """
    return _block(x, labels, out_channels, num_classes, "c_" + name, ops.conv2d, kernel, strides, dilations, act,
                  pooling, padding)


def up_sampling(x, pooling, out_channels, conv, kernel, strides, name, padding='SAME'):
    """
      Performs upsampling on given input. The upsampling factor for height and width is determined by stride tuple.
      There are 3 types of upsampling supported:
        - Nearest neighbour (in there it is called avg)
        - Deconvolutional (in there it is called conv)
        - Subpixel shuffle (in there it is called subpixel)
    Args:
      x: The input vector.
      pooling: Strategy of pooling. Options: avg, conv, subpixel
      out_channels: Number of features in the output layer.
      conv: Convolution function. Options conv2d or snconv2d
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      name: Scope name
      padding:  Type of padding  (Default value = 'SAME')

    Returns:
      Returns input with increased dimentions.
    """
    if pooling == 'avg':
        x = up_sample(x, strides)
        x = conv(x, out_channels, kernel, name=name, padding=padding)
    elif pooling == 'conv':
        new_shape = [x.get_shape().as_list()[0], x.get_shape().as_list()[1] * strides[0],
                     x.get_shape().as_list()[2] * strides[1],
                     int(out_channels)]
        if conv == ops.snconv2d:
            x = ops.sndeconv2d(x, new_shape, kernel, strides, name='de' + name)
        elif conv == ops.scaled_conv2d:
            x = ops.scaled_deconv2d(x, new_shape, kernel, strides, name='de' + name)
        else:
            x = ops.deconv2d(x, new_shape, kernel, strides, name='de' + name)
    elif pooling == 'subpixel':
        x = upsample_ps(x, strides)
        x = conv(x, out_channels, kernel, name=name, padding=padding)
    elif pooling == "None":
        x = conv(x, out_channels, (1, 1), name=name, padding=padding)
    return x


def get_kernel(x, kernel):
    """
    Calculates the kernel size given the input. Kernel size is changed only if the input dimentions are smaller
    than kernel

    Args:
      x: The input vector.
      kernel:  The height and width of the convolution kernel filter

    Returns:
      The height and width of new convolution kernel filter
    """
    height = kernel[0]
    width = kernel[1]
    if x.get_shape().as_list()[1] < height:
        height = x.get_shape().as_list()[1]
    elif x.get_shape().as_list()[2] < width:
        width = x.get_shape().as_list()[2]

    return (height, width)


def get_dimentions_factors(strides_schedule):
    """
      Method calculates how much height and width will be increased/decreased basen on given stride schedule
    Args:
      strides_schedule: A list of stride tuples(heigh, width)

    Returns:
      Two numbers that tells how many times height and width will be increased/decreased
    """
    width_d = 1
    height_d = 1
    for s in strides_schedule:
        width_d = width_d * s[1]
        height_d = height_d * s[0]
    return height_d, width_d
