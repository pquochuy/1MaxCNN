%% this setting similar to those in  McLoughlin I, Zhang H.-M., Xie Z.-P., Song Y., Xiao. W., “Robust Sound Event Classification using Deep Neural Networks”, IEEE Trans. Audio, Speech and Language Processing, Jan 2015
%% which can be found here http://www.lintech.org/machine_hearing/index.html

clear all
close all
clc

% load config
config;

%%clean signals
train_data = cell(nclass*40,1);
train_label = zeros(nclass*40,1);
train_y = zeros(nclass*40,nclass); % one-hot format

file_list = {};
ntrain = 0;
for class=1 : nclass
    sub_d=dir([wave_dir,'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) > 3 %select the specific files used for training
            file_list = [file_list; [wave_dir,'/',class_name{class},'/',sub_d(file+2).name]];
            ntrain = ntrain + 1;
            train_label(ntrain) = class;
            train_y(ntrain,train_label(ntrain)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
	disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
	train_data{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% dev clean signals
dev_data = cell(nclass*10,1);
dev_label = zeros(nclass*10,1);
dev_y = zeros(nclass*10,nclass); % one-hot format

file_list = {};
ndev=0;
for class=1 : nclass
    sub_d=dir([wave_dir,'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) == 3 %select the specific files used for training
            file_list = [file_list; [wave_dir,'/',class_name{class},'/',sub_d(file+2).name]];
            ndev = ndev + 1;
            dev_label(ndev) = class;
            dev_y(ndev,dev_label(ndev)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
    dev_data{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(train_data)
    mi = min(mi, min(min(train_data{i})));
end
for i = 1 : numel(dev_data)
    mi = min(mi, min(min(dev_data{i})));
end
for i = 1 : numel(train_data)
    train_data{i} = train_data{i} - mi;
end
for i = 1 : numel(dev_data)
    dev_data{i} = dev_data{i} - mi;
end
for i = 1 : numel(train_data)
    ma = max(ma, max(max(train_data{i})));
end
for i = 1 : numel(dev_data)
    ma = max(ma, max(max(dev_data{i})));
end
for i = 1 : numel(train_data)
    train_data{i} = train_data{i}/ma;
end
for i = 1 : numel(dev_data)
    dev_data{i} = dev_data{i}/ma;
end
save('train_data_plus.mat','train_data','train_label','train_y');
save('dev_data_plus.mat','dev_data','dev_label','dev_y');


% test clean signals
test_data = cell(nclass*30,1);
test_label = zeros(nclass*30,1);
test_y = zeros(nclass*30,nclass); % one-hot format

file_list = {};
ntest=0;
for class=1 : nclass
    sub_d=dir([wave_dir,'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) < 3 %select the specific files used for training
            file_list = [file_list; [wave_dir,'/',class_name{class},'/',sub_d(file+2).name]];
            ntest = ntest + 1;
            test_label(ntest) = class;
            test_y(ntest,test_label(ntest)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
    [wave,fs]=audioread(file_list{i});
    test_data{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(test_data)
    mi = min(mi, min(min(test_data{i})));
end
for i = 1 : numel(test_data)
    test_data{i} = test_data{i} - mi;
end
for i = 1 : numel(test_data)
    ma = max(ma, max(max(test_data{i})));
end
for i = 1 : numel(test_data)
    test_data{i} = test_data{i}/ma;
end
save('test_data_plus.mat','test_data','test_label','test_y');

%% mix0 signals
train_data_mix0 = cell(nclass*40,1);
train_label_mix0 = zeros(nclass*40,1);
train_y_mix0 = zeros(nclass*40,nclass); % one-hot format

file_list = {};
ntrain = 0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{1},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) > 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{1},'/',class_name{class},'/',sub_d(file+2).name]];
            ntrain = ntrain + 1;
            train_label_mix0(ntrain) = class;
            train_y_mix0(ntrain,train_label_mix0(ntrain)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
	disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
	train_data_mix0{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% dev mix0 signals
dev_data_mix0 = cell(nclass*10,1);
dev_label_mix0 = zeros(nclass*10,1);
dev_y_mix0 = zeros(nclass*10,nclass); % one-hot format

file_list = {};
ndev=0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{1},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) == 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{1},'/',class_name{class},'/',sub_d(file+2).name]];
            ndev = ndev + 1;
            dev_label_mix0(ndev) = class;
            dev_y_mix0(ndev,dev_label_mix0(ndev)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
    dev_data_mix0{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(train_data_mix0)
    mi = min(mi, min(min(train_data_mix0{i})));
end
for i = 1 : numel(dev_data_mix0)
    mi = min(mi, min(min(dev_data_mix0{i})));
end
for i = 1 : numel(train_data_mix0)
    train_data_mix0{i} = train_data_mix0{i} - mi;
end
for i = 1 : numel(dev_data_mix0)
    dev_data_mix0{i} = dev_data_mix0{i} - mi;
end
for i = 1 : numel(train_data_mix0)
    ma = max(ma, max(max(train_data_mix0{i})));
end
for i = 1 : numel(dev_data_mix0)
    ma = max(ma, max(max(dev_data_mix0{i})));
end
for i = 1 : numel(train_data_mix0)
    train_data_mix0{i} = train_data_mix0{i}/ma;
end
for i = 1 : numel(dev_data_mix0)
    dev_data_mix0{i} = dev_data_mix0{i}/ma;
end
save('train_data_mix0_plus.mat','train_data_mix0','train_label_mix0','train_y_mix0');
save('dev_data_mix0_plus.mat','dev_data_mix0','dev_label_mix0','dev_y_mix0');


% test mix0 signals
test_data_mix0 = cell(nclass*30,1);
test_label_mix0 = zeros(nclass*30,1);
test_y_mix0 = zeros(nclass*30,nclass); % one-hot format

file_list = {};
ntest=0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{1},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) < 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{1},'/',class_name{class},'/',sub_d(file+2).name]];
            ntest = ntest + 1;
            test_label_mix0(ntest) = class;
            test_y_mix0(ntest,test_label_mix0(ntest)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
    [wave,fs]=audioread(file_list{i});
    test_data_mix0{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(test_data_mix0)
    mi = min(mi, min(min(test_data_mix0{i})));
end
for i = 1 : numel(test_data_mix0)
    test_data_mix0{i} = test_data_mix0{i} - mi;
end
for i = 1 : numel(test_data_mix0)
    ma = max(ma, max(max(test_data_mix0{i})));
end
for i = 1 : numel(test_data_mix0)
    test_data_mix0{i} = test_data_mix0{i}/ma;
end
save('test_data_mix0_plus.mat','test_data_mix0','test_label_mix0','test_y_mix0');


%% mix10 signals
train_data_mix10 = cell(nclass*40,1);
train_label_mix10 = zeros(nclass*40,1);
train_y_mix10 = zeros(nclass*40,nclass); % one-hot format

file_list = {};
ntrain = 0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{2},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) > 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{2},'/',class_name{class},'/',sub_d(file+2).name]];
            ntrain = ntrain + 1;
            train_label_mix10(ntrain) = class;
            train_y_mix10(ntrain,train_label_mix10(ntrain)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
	disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
	train_data_mix10{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% dev mix10 signals
dev_data_mix10 = cell(nclass*10,1);
dev_label_mix10 = zeros(nclass*10,1);
dev_y_mix10 = zeros(nclass*10,nclass); % one-hot format

file_list = {};
ndev=0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{2},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) == 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{2},'/',class_name{class},'/',sub_d(file+2).name]];
            ndev = ndev + 1;
            dev_label_mix10(ndev) = class;
            dev_y_mix10(ndev,dev_label_mix10(ndev)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
    dev_data_mix10{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(train_data_mix10)
    mi = min(mi, min(min(train_data_mix10{i})));
end
for i = 1 : numel(dev_data_mix10)
    mi = min(mi, min(min(dev_data_mix10{i})));
end
for i = 1 : numel(train_data_mix10)
    train_data_mix10{i} = train_data_mix10{i} - mi;
end
for i = 1 : numel(dev_data_mix10)
    dev_data_mix10{i} = dev_data_mix10{i} - mi;
end
for i = 1 : numel(train_data_mix10)
    ma = max(ma, max(max(train_data_mix10{i})));
end
for i = 1 : numel(dev_data_mix10)
    ma = max(ma, max(max(dev_data_mix10{i})));
end
for i = 1 : numel(train_data_mix10)
    train_data_mix10{i} = train_data_mix10{i}/ma;
end
for i = 1 : numel(dev_data_mix10)
    dev_data_mix10{i} = dev_data_mix10{i}/ma;
end
save('train_data_mix10_plus.mat','train_data_mix10','train_label_mix10','train_y_mix10');
save('dev_data_mix10_plus.mat','dev_data_mix10','dev_label_mix10','dev_y_mix10');


% test mix10 signals
test_data_mix10 = cell(nclass*30,1);
test_label_mix10 = zeros(nclass*30,1);
test_y_mix10 = zeros(nclass*30,nclass); % one-hot format

file_list = {};
ntest=0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{2},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) < 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{2},'/',class_name{class},'/',sub_d(file+2).name]];
            ntest = ntest + 1;
            test_label_mix10(ntest) = class;
            test_y_mix10(ntest,test_label_mix10(ntest)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
    [wave,fs]=audioread(file_list{i});
    test_data_mix10{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(test_data_mix10)
    mi = min(mi, min(min(test_data_mix10{i})));
end
for i = 1 : numel(test_data_mix10)
    test_data_mix10{i} = test_data_mix10{i} - mi;
end
for i = 1 : numel(test_data_mix10)
    ma = max(ma, max(max(test_data_mix10{i})));
end
for i = 1 : numel(test_data_mix10)
    test_data_mix10{i} = test_data_mix10{i}/ma;
end
save('test_data_mix10_plus.mat','test_data_mix10','test_label_mix10','test_y_mix10');


%% mix20 signals
train_data_mix20 = cell(nclass*40,1);
train_label_mix20 = zeros(nclass*40,1);
train_y_mix20 = zeros(nclass*40,nclass); % one-hot format

file_list = {};
ntrain = 0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{3},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) > 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{3},'/',class_name{class},'/',sub_d(file+2).name]];
            ntrain = ntrain + 1;
            train_label_mix20(ntrain) = class;
            train_y_mix20(ntrain,train_label_mix20(ntrain)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
	disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
	train_data_mix20{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% dev mix20 signals
dev_data_mix20 = cell(nclass*10,1);
dev_label_mix20 = zeros(nclass*10,1);
dev_y_mix20 = zeros(nclass*10,nclass); % one-hot format

file_list = {};
ndev=0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{3},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) == 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{3},'/',class_name{class},'/',sub_d(file+2).name]];
            ndev = ndev + 1;
            dev_label_mix20(ndev) = class;
            dev_y_mix20(ndev,dev_label_mix20(ndev)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
	[wave,fs]=audioread(file_list{i});
    dev_data_mix20{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(train_data_mix20)
    mi = min(mi, min(min(train_data_mix20{i})));
end
for i = 1 : numel(dev_data_mix20)
    mi = min(mi, min(min(dev_data_mix20{i})));
end
for i = 1 : numel(train_data_mix20)
    train_data_mix20{i} = train_data_mix20{i} - mi;
end
for i = 1 : numel(dev_data_mix20)
    dev_data_mix20{i} = dev_data_mix20{i} - mi;
end
for i = 1 : numel(train_data_mix20)
    ma = max(ma, max(max(train_data_mix20{i})));
end
for i = 1 : numel(dev_data_mix20)
    ma = max(ma, max(max(dev_data_mix20{i})));
end
for i = 1 : numel(train_data_mix20)
    train_data_mix20{i} = train_data_mix20{i}/ma;
end
for i = 1 : numel(dev_data_mix20)
    dev_data_mix20{i} = dev_data_mix20{i}/ma;
end
save('train_data_mix20_plus.mat','train_data_mix20','train_label_mix20','train_y_mix20');
save('dev_data_mix20_plus.mat','dev_data_mix20','dev_label_mix20','dev_y_mix20');


% test mix20 signals
test_data_mix20 = cell(nclass*30,1);
test_label_mix20 = zeros(nclass*30,1);
test_y_mix20 = zeros(nclass*30,nclass); % one-hot format

file_list = {};
ntest=0;
for class=1 : nclass
    sub_d=dir([wave_dir_mix{3},'/',class_name{class}]);
    for file = 1 : nfile
        if mod(file - 1,8) < 3 %select the specific files used for training
            file_list = [file_list; [wave_dir_mix{3},'/',class_name{class},'/',sub_d(file+2).name]];
            ntest = ntest + 1;
            test_label_mix20(ntest) = class;
            test_y_mix20(ntest,test_label_mix20(ntest)) = 1;
        end
    end
end

% comment out for paralell loop
%parfor i = 1 : numel(file_list)
for i = 1 : numel(file_list)
    disp(file_list{i});
    [wave,fs]=audioread(file_list{i});
    test_data_mix20{i} = SIF_plus(wave,fs,ny,winlen,overlap);
end

% normalization to [0,1]
mi = Inf;
ma = 0.0;
for i = 1 : numel(test_data_mix20)
    mi = min(mi, min(min(test_data_mix20{i})));
end
for i = 1 : numel(test_data_mix20)
    test_data_mix20{i} = test_data_mix20{i} - mi;
end
for i = 1 : numel(test_data_mix20)
    ma = max(ma, max(max(test_data_mix20{i})));
end
for i = 1 : numel(test_data_mix20)
    test_data_mix20{i} = test_data_mix20{i}/ma;
end
save('test_data_mix20_plus.mat','test_data_mix20','test_label_mix20','test_y_mix20');