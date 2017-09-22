#CNN with CIFAR


import Getting_data
import tensorflow as tf
import numpy as np
from CNN_layers import *

training_data, labels = Getting_data.creating_dataset()


#labels shape:  (50000, 10)
#training data shape:  (50000, 32, 32, 3)


class CreateBatches(object):
    def __init__(self):
        self.index = 0


    def next_batch(self, data,labels, index, batch_size):
        batch_data = data[self.index:self.index+batch_size]
        batch_labels = labels[self.index:self.index+batch_size]
        
        self.index += batch_size

        return batch_data, batch_labels
        
    

BATCH_SIZE = 50
TOTAL_BATCHES = int(len(training_data)/BATCH_SIZE)
EPOCHS = 1



x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)





conv1 = conv_layer(x, shape =[5,5,3,32])
conv1_pool = max_pool_2x2(conv1)

#conv1_pool_shape =tf.Print (conv1_pool.get_shape())


conv2 = conv_layer(conv1_pool, shape = [5,5,32,64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool,  [-1, 8*8*64])


#conv2_shape = (conv2.get_shape())
#conv2_pool_shape = (conv2_pool.get_shape())

#conv2_flat_shape = conv2_flat.get_shape()



full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob = keep_prob)
y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =y_conv, labels =y))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), (tf.argmax(y, 1)))
accuracy =( tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(EPOCHS):
        Batches = CreateBatches()
        for _ in range(3000):
            batch_data, batch_label =Batches.next_batch(training_data , labels, Batches.index, BATCH_SIZE)
            sess.run(train_step, feed_dict = {x: batch_data, y:batch_label, keep_prob: 0.5})
            
            
        #acc_score = sess.run(accuracy, feed_dict = {x: training_data[-300:-1], y:labels[-300: -1], keep_prob: 1.0})
        #print(acc_score)
            
        














