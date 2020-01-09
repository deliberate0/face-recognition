import tensorflow as tf
from BuildNet import NetworkBuilder
from prepareData import DataSetGenerator
import datetime
import numpy as np
import os
import cv2

with tf.name_scope("Input") as scope:
    input_img = tf.placeholder(dtype='float32', shape=[None, 128, 128, 3], name="input")

with tf.name_scope("Target") as scope:
    target_labels = tf.placeholder(dtype='float32', shape=[None, 10], name="Targets")


netBuild = NetworkBuilder()

with tf.name_scope("Modelv2") as scope:
    model = input_img
    model = netBuild.attach_conv_layer(model, 32, summary=True)
    model = netBuild.attach_relu_layer(model)
    model = netBuild.attach_conv_layer(model, 32, summary=True)
    model = netBuild.attach_relu_layer(model)
    model = netBuild.attach_pooling_layer(model)

    model = netBuild.attach_conv_layer(model, 64, summary=True)
    model = netBuild.attach_relu_layer(model)
    model = netBuild.attach_conv_layer(model, 64, summary=True)
    model = netBuild.attach_relu_layer(model)
    model = netBuild.attach_pooling_layer(model)

    model = netBuild.attach_conv_layer(model, 128, summary=True)
    model = netBuild.attach_relu_layer(model)
    model = netBuild.attach_conv_layer(model, 128, summary=True)
    model = netBuild.attach_relu_layer(model)
    model = netBuild.attach_pooling_layer(model)

    model = netBuild.flatten(model)
    model = netBuild.attach_dense_layer(model, 200, summary=True)
    model = netBuild.attach_sigmod_layer(model)
    model = netBuild.attach_dense_layer(model, 32, summary=True)
    model = netBuild.attach_sigmod_layer(model)
    model = netBuild.attach_dense_layer(model, 10)

    prediction = netBuild.attach_soft_max_layer(model)

# optimization and accuray

with tf.name_scope("Optimization") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target_labels)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost", cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#test
test_img=[]
for filename in os.listdir('test'):
    img = cv2.imread(os.path.join('test',filename))
    test_img.append(img)

labels=[]
for x in range(13):
    labels.append(list([0,0,0,0,0,0,0,0,0,1]))

labels=np.array(labels)

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint("saved-model-v2/"))
    accuracy = sess.run([accuracy],feed_dict={input_img:test_img,target_labels:labels})
    for x in accuracy:
        print(x)