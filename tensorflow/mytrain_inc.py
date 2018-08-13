#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from shutil import copyfile
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
#tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
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
tf.flags.DEFINE_string("out_dir", "./output", "Point to output directory")


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
data_path = os.path.abspath(FLAGS.train_data)
data = loadmat(data_path)
x_train = data['train_data']
y_train = data['train_y']
label_train = data['train_label']
data_path = os.path.abspath(FLAGS.test_data)
data = loadmat(data_path)
x_test = data['dev_data']
y_test = data['dev_y']
label_test = data['dev_label']

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(label_train)))
x_train_shuffled = x_train[shuffle_indices]
y_train_shuffled = y_train[shuffle_indices]
label_train_shuffled = label_train[shuffle_indices]

#expand dim
x_train_shuffled = np.expand_dims(x_train_shuffled,axis=3)
x_test = np.expand_dims(x_test,axis=3)
print("Train/Test set: {:d}/{:d}".format(len(label_train), len(label_test)))

max_acc = 0.0

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = AEDCNN(
            time_length=x_train.shape[1],
            freq_length=x_train.shape[2],
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

        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(out_dir, "best_model")
        if os.path.isfile(best_dir):
            saver.restore(sess, best_dir)
            print("Model loaded")
        else:
            print("Model initialized")
            sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

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
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                acc = acc + accuracy
                if writer:
                    writer.add_summary(summaries, step)
            return acc/N
                

        # Generate batches
        batches = data_helpers.batch_iter(
            zip(x_train_shuffled, y_train_shuffled), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                acc = dev_step(x_test, y_test, writer=dev_summary_writer)
                print("")
                # my log file
                print("Average accuracy: {:g}".format(acc))
                with open(os.path.join(out_dir,"acc_log.txt"), "a") as text_file:
                    text_file.write("{0}\n".format(acc))
                
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
            
                if acc > max_acc:
                    max_acc = acc
                    best_dir = os.path.join(out_dir, "best_model")
                    copyfile(path,best_dir)
                    print("Best model copied in file: {}\n".format(best_dir))



