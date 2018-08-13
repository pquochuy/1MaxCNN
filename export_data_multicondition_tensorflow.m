clear all
close all
clc

load('train_data_tensorflow.mat');
load('dev_data_tensorflow.mat');
load('test_data_tensorflow.mat');

load('train_data_mix0_tensorflow.mat');
load('dev_data_mix0_tensorflow.mat');
load('test_data_mix0_tensorflow.mat');

load('train_data_mix10_tensorflow.mat');
load('dev_data_mix10_tensorflow.mat');
load('test_data_mix10_tensorflow.mat');

load('train_data_mix20_tensorflow.mat');
load('dev_data_mix20_tensorflow.mat');
load('test_data_mix20_tensorflow.mat');

train_data = [train_data; train_data_mix20; train_data_mix10; train_data_mix0];
train_label = [train_label; train_label_mix20; train_label_mix10; train_label_mix0];
train_y = [train_y; train_y_mix20; train_y_mix10; train_y_mix0];
save('train_data_multi_tensorflow.mat','train_data','train_label','train_y');

dev_data = [dev_data; dev_data_mix20; dev_data_mix10; dev_data_mix0];
dev_label = [dev_label; dev_label_mix20; dev_label_mix10; dev_label_mix0];
dev_y = [dev_y; dev_y_mix20; dev_y_mix10; dev_y_mix0];
save('dev_data_multi_tensorflow.mat','dev_data','dev_label','dev_y');

test_data = [test_data; test_data_mix20; test_data_mix10; test_data_mix0];
test_label = [test_label; test_label_mix20; test_label_mix10; test_label_mix0];
test_y = [test_y; test_y_mix20; test_y_mix10; test_y_mix0];
save('test_data_multi_tensorflow.mat','test_data','test_label','test_y');


