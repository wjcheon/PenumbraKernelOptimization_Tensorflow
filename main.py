import tensorflow as tf
import numpy as np
import scipy.io as sio
import csv
#import matplotlib.pyplot as plt
import pylab as plt
##
scint_img = list(csv.reader(open('scint_registered.csv')))
scint_img = np.asanyarray(scint_img, np.float)
#
film_img = list(csv.reader(open('film_registered.csv')))
film_img = np.asanyarray(film_img, np.float)
##
plt.figure()
plt.imshow(scint_img)
plt.figure()
plt.imshow(film_img)
#
scint_img_input = np.expand_dims(scint_img, axis= -1)
scint_img_input = np.expand_dims(scint_img_input, axis = 0)
film_img_input = np.expand_dims(film_img, axis=-1)
film_img_input = np.expand_dims(film_img_input, axis = 0)
shape_img = np.shape(film_img_input)
## Tensorflow
#hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
filter_size = 100
#
# input place holders
X = tf.placeholder(tf.float32, [None, shape_img[1], shape_img[2], 1])
X_img = tf.reshape(X, [-1, shape_img[1], shape_img[2], 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, shape_img[1], shape_img[2], 1])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([filter_size, filter_size, 1, 1], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
logits = L1

cost = tf.reduce_mean(tf.losses.mean_squared_error(Y, logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
##
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    batch_xs = scint_img_input
    batch_ys = film_img_input
    feed_dict = {X: batch_xs, Y: batch_ys}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost = c

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

##
logits_val = sess.run(logits, feed_dict=feed_dict)
# logits
logits_val_squeeze = np.squeeze(logits_val)
np.shape(logits_val_squeeze)
line_profile = logits_val_squeeze[245,:]
# Y
Y_val_squeeze = np.squeeze(film_img_input)
np.shape(Y_val_squeeze)
line_profile_Y = Y_val_squeeze[245, :]
##
plt.figure()
plt.plot(line_profile, 'b--')
plt.plot(line_profile_Y, 'r')