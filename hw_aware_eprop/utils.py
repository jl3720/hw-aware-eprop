import torch
import numpy as np

def exp_convolve(tensor, decay):
    '''
    Filters a tensor with an exponential filter.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param decay: a decay constant of the form exp(-dt/tau) with tau the time constant
    :return: the filtered tensor of shape (trial, time, neuron)
    '''
    #with tf.name_scope('ExpConvolve'):
    #    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    r_shp = range(len(tensor.size()))
    transpose_perm = [1, 0] + list(r_shp)[2:]

    tensor_time_major = tensor.permute(transpose_perm)
    initializer = torch.zeros_like(tensor_time_major[0])
    #filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
    #########################################################################
    exp_kernel = lambda a, x: a * decay + (1 - decay) * x
    filtered_tensor = []
    tensor_upd = exp_kernel( initializer, tensor_time_major[0] )
    filtered_tensor.append( tensor_upd )
    for i in range( 1, tensor_time_major.size(0) ):
        tensor_upd = exp_kernel( tensor_upd, tensor_time_major[i] )
        filtered_tensor.append( tensor_upd )
    filtered_tensor = torch.stack( filtered_tensor, dim=0 )
    #########################################################################
    filtered_tensor = filtered_tensor.permute(transpose_perm)

    return filtered_tensor

def shift_by_one_time_step(tensor, initializer=None):
    '''
    Shift the input on the time dimension by one.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param initializer: pre-prend this as the new first element on the time dimension
    :return: a shifted tensor of shape (trial, time, neuron)
    
    note: initializer has to be shaped as [trian, neuron], as it will be automatically
    unsqueezed inside the function
    '''
    #with tf.name_scope('TimeShift'):
    #    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    r_shp = range(len(tensor.size()))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    tensor_time_major = tensor.permute(transpose_perm)

    if initializer is None:
        initializer = torch.zeros_like(tensor_time_major[0])

    shifted_tensor = torch.cat([initializer.unsqueeze(0), tensor_time_major[:-1]], dim=0)

    shifted_tensor = shifted_tensor.permute(transpose_perm)
    return shifted_tensor