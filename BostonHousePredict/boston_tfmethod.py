import tensorflow as tf
import numpy as np
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


X=tf.placeholder('float32',[None,13])
y=tf.placeholder('float32',[None])

keep_prob=tf.placeholder('float32')

normal1=tf.layers.batch_normalization(X)

reshape1=tf.reshape(normal1,shape=(-1,13,1,1))
conv1_1=tf.layers.conv2d(reshape1,filters=13,strides=1,padding='same',kernel_size=1,activation=tf.nn.sigmoid)
conv1_2=tf.layers.conv2d(conv1_1,filters=26,strides=2,padding='same',kernel_size=2,activation=tf.nn.sigmoid)
pool1=tf.layers.max_pooling2d(conv1_2,pool_size=2,strides=1,padding='same')

conv2_1=tf.layers.conv2d(pool1,filters=52,strides=1,padding='same',kernel_size=1,activation=tf.nn.sigmoid)
conv2_2=tf.layers.conv2d(conv2_1,filters=104,strides=2,padding='same',kernel_size=2,activation=tf.nn.sigmoid)
pool2=tf.layers.max_pooling2d(conv2_2,pool_size=2,strides=1,padding='same')

reshape2=tf.reshape(pool2,shape=(-1,416))
dropout1=tf.nn.dropout(reshape2,keep_prob=keep_prob)

dense1=tf.layers.dense(dropout1,units=1)
reshape3=tf.reshape(dense1,shape=(-1,))

loss=tf.reduce_mean(tf.abs(y-reshape3))
step=tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(30000):
        sess.run(step,feed_dict={X:x_train,y:y_train,keep_prob:0.5})
        # print(loss_val)

        if i % 1000 == 0:
            loss_val=sess.run(loss,feed_dict={X:x_test,y:y_test,keep_prob:1})
            print(str(i)+"=>"+str(loss_val))