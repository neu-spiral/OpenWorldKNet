from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


class KernelLayer(Layer):

    def __init__(self, batch_size, sigma, **kwargs):
        self.batch_size = batch_size
        self.sigma = sigma
        super(KernelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(KernelLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # compute kernel matrix
        bs = self.batch_size
        sigma = self.sigma
        # kernel = tf.get_variable(name='kernel', shape=(bs, bs)) 
        kernel_list = []
        for i in range(bs):
            dif = x[i, :] - x
            kernel_list.append( K.exp(-K.sum(dif*dif, axis=1) / (2*sigma*sigma)) )
        kernel = tf.stack(kernel_list, axis=1)
        # normalization
        ds = 1.0 / K.sqrt(K.sum(kernel, axis=0))
        D = tf.einsum('i,j->ij', ds, ds) # outer product
        n_kernel = kernel * D
        return n_kernel

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0])

    def set_sigma(self, sigma):
        self.sigma = sigma