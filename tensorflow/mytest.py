#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from scipy.io import loadmat
from aed_cnn import AEDCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "5,10,15,20,25,30,35,40,45", "Comma-separated filter sizes (default: '1,3,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.flags.DEFINE_string("train_data", "../data/train_data_1.mat", "Point to directory of input data")
tf.flags.DEFINE_string("test_data", "../data/test_data_1.mat", "Point to directory of input data")
tf.flags.DEFINE_string("test_data_mix20", "../data/test_data_mix20_1.mat", "Point to directory of input data")
tf.flags.DEFINE_string("test_data_mix10", "../data/test_data_mix10_1.mat", "Point to directory of input data")
tf.flags.DEFINE_string("test_data_mix0", "../data/test_data_mix0_1.mat", "Point to directory of input data")
tf.flags.DEFINE_string("out_dir", "runs/ny_64", "Point to output directory")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
data_path = os.path.abspath(FLAGS.test_data)
data = loadmat(data_path)
x_test = data['test_data']
y_test = data['test_y']
label_test = data['test_label']

data_path = os.path.abspath(FLAGS.test_data_mix20)
data = loadmat(data_path)
x_test_mix20 = data['test_data_mix20']
y_test_mix20 = data['test_y_mix20']
label_test_mix20 = data['test_label_mix20']

data_path = os.path.abspath(FLAGS.test_data_mix10)
data = loadmat(data_path)
x_test_mix10 = data['test_data_mix10']
y_test_mix10 = data['test_y_mix10']
label_test_mix10 = data['test_label_mix10']

data_path = os.path.abspath(FLAGS.test_data_mix0)
data = loadmat(data_path)
x_test_mix0 = data['test_data_mix0']
y_test_mix0 = data['test_y_mix0']
label_test_mix0 = data['test_label_mix0']

# Randomly shuffle data
np.random.seed(10)

#expand dim
x_test = np.expand_dims(x_test,axis=3)
x_test_mix20 = np.expand_dims(x_test_mix20,axis=3)
x_test_mix10 = np.expand_dims(x_test_mix10,axis=3)
x_test_mix0 = np.expand_dims(x_test_mix0,axis=3)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = AEDCNN(
            time_length=x_test.shape[1],
            freq_length=x_test.shape[2],
            num_classes=50,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        
        filter_sizes=map(int, FLAGS.filter_sizes.split(","))
        str_dir = ""
        for i in range(len(filter_sizes)-1):
            str_dir = str_dir + "{:d}".format(filter_sizes[i]) + "_"
        str_dir = str_dir + "{:d}".format(filter_sizes[-1])
        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        out_dir = os.path.abspath(os.path.join(out_dir, str_dir))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        #sess.run(tf.initialize_all_variables())
        best_dir = os.path.join(out_dir, "best_model")		
        saver.restore(sess, best_dir)
        print("Model loaded")
        
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            Ntest = 100 # test batch by batch of Ntest samples due to memory issue
            acc = 0.0
            N = len(x_batch)/Ntest
            for i in range(N):            
                x_ = x_batch[i*Ntest : (i+1)*Ntest]
                y_ = y_batch[i*Ntest : (i+1)*Ntest]
                feed_dict = {
                  cnn.input_x: x_,
                  cnn.input_y: y_,
                  cnn.dropout_keep_prob: 1.0
                }
                #yhat, accuracy = sess.run([cnn.predicions, cnn.accuracy], feed_dict)
                accuracy = sess.run(cnn.accuracy, feed_dict)
                print("accuracy {:g}".format(accuracy))
                acc = acc + accuracy
            return acc/N

        print("Test clean signals")
        acc = dev_step(x_test, y_test)
        print("")
        # my log file
        print("Accuracy clean: {:g}".format(acc))	
        with open(os.path.join(out_dir,"test_acc.txt"), "a") as text_file:
            text_file.write("Test clean signals: {0}\n".format(acc))

        print("Test mix20 signals")
        acc_mix20 = dev_step(x_test_mix20, y_test_mix20)
        print("")
        # my log file
        print("Average accuracy mix20: {:g}".format(acc_mix20))	
        with open(os.path.join(out_dir,"test_acc.txt"), "a") as text_file:
            text_file.write("Test mix20 signals: {0}\n".format(acc_mix20))

        print("Test mix10 signals")
        acc_mix10 = dev_step(x_test_mix10, y_test_mix10)
        print("")
        # my log file
        print("Average accuracy mix10: {:g}".format(acc_mix10))	
        with open(os.path.join(out_dir,"test_acc.txt"), "a") as text_file:
            text_file.write("Test mix10 signals: {0}\n".format(acc_mix10))

        print("Test mix0 signals")
        acc_mix0 = dev_step(x_test_mix0, y_test_mix0)
        print("")
        # my log file
        print("Average accuracy mix0: {:g}".format(acc_mix0))	
        with open(os.path.join(out_dir,"test_acc.txt"), "a") as text_file:
            text_file.write("Test mix0 signals: {0}\n".format(acc_mix0))
