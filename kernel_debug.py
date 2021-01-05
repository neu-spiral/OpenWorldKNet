import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

tf.enable_eager_execution()

x_np = np.random.randn(128, 32)

bs = 128
sigma = 1
x = tf.convert_to_tensor(x_np, dtype=tf.float32)
# kernel = tf.get_variable(name='kernel', shape=(bs, bs)) 
kernel_list = []

print(x[0, :] - x)
print(x_np[0, :] - x_np)

for i in range(bs):
    dif = x[i, :] - x
    kernel_list.append( tf.exp(-tf.math.reduce_sum(dif*dif, axis=1) / (2*sigma*sigma)) )
kernel = tf.stack(kernel_list, axis=1)
# normalization
ds = 1.0 / tf.sqrt(tf.math.reduce_sum(kernel, axis=0))
D = tf.einsum('i,j->ij', ds, ds) # outer product
n_kernel = kernel * D

print(kernel)
print(rbf_kernel(x_np, gamma= 1.0/(2*sigma*sigma)))