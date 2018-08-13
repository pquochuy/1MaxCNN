clear all
close all
clc

load('train_data_plus.mat');
load('dev_data_plus.mat');
load('test_data_plus.mat');
load('train_data_mix0_plus.mat');
load('dev_data_mix0_plus.mat');
load('test_data_mix0_plus.mat');
load('train_data_mix10_plus.mat');
load('dev_data_mix10_plus.mat');
load('test_data_mix10_plus.mat');
load('train_data_mix20_plus.mat');
load('dev_data_mix20_plus.mat');
load('test_data_mix20_plus.mat');

max_length = 0;
for i =1 : numel(train_data)
    max_length = max(max_length,size(train_data{i},2));
end
for i =1 : numel(dev_data)
    max_length = max(max_length,size(dev_data{i},2));
end
for i =1 : numel(test_data)
    max_length = max(max_length,size(test_data{i},2));
end

% padding zeros
%parfor i =1 : numel(train_data)
for i =1 : numel(train_data)
    train_data{i} = transpose(padarray(train_data{i},[0, max_length - size(train_data{i},2)],'post'));
end
tmp = zeros(numel(train_data),size(train_data{1},1), size(train_data{1},2));
%parfor i =1 : numel(train_data)
for i =1 : numel(train_data)
    tmp(i,:,:) = train_data{i};
end
train_data = tmp;
clear tmp
train_data = single(train_data); % convert to single precision (float 32)
train_label = single(train_label); % convert to single precision (float 32)
train_y = single(train_y); % convert to single precision (float 32)
save('train_data_tensorflow_plus.mat','train_data','train_label','train_y');

%parfor i =1 : numel(dev_data)
for i =1 : numel(dev_data)
    dev_data{i} = transpose(padarray(dev_data{i},[0, max_length - size(dev_data{i},2)],'post'));
end
tmp = zeros(numel(dev_data),size(dev_data{1},1), size(dev_data{1},2));
%parfor i =1 : numel(dev_data)
for i =1 : numel(dev_data)
    tmp(i,:,:) = dev_data{i};
end
dev_data = tmp;
clear tmp
dev_data = single(dev_data); % convert to single precision (float 32)
dev_label = single(dev_label); % convert to single precision (float 32)
dev_y = single(dev_y); % convert to single precision (float 32).
save('dev_data_tensorflow_plus.mat','dev_data','dev_label','dev_y');

%parfor i =1 : numel(test_data)
for i =1 : numel(test_data)
    test_data{i} = transpose(padarray(test_data{i},[0, max_length - size(test_data{i},2)],'post'));
end
tmp = zeros(numel(test_data),size(test_data{1},1), size(test_data{1},2));
%parfor i =1 : numel(test_data)
for i =1 : numel(test_data)
    tmp(i,:,:) = test_data{i};
end
test_data = tmp;
clear tmp
test_data = single(test_data); % convert to single precision (float 32)
test_label = single(test_label); % convert to single precision (float 32)
test_y = single(test_y); % convert to single precision (float 32).
save('test_data_tensorflow_plus.mat','test_data','test_label','test_y');



% padding zeros
%parfor i =1 : numel(train_data_mix0)
for i =1 : numel(train_data_mix0)
    train_data_mix0{i} = transpose(padarray(train_data_mix0{i},[0, max_length - size(train_data_mix0{i},2)],'post'));
end
tmp = zeros(numel(train_data_mix0),size(train_data_mix0{1},1), size(train_data_mix0{1},2));
%parfor i =1 : numel(train_data_mix0)
for i =1 : numel(train_data_mix0)
    tmp(i,:,:) = train_data_mix0{i};
end
train_data_mix0 = tmp;
clear tmp
train_data_mix0 = single(train_data_mix0); % convert to single precision (float 32)
train_label_mix0 = single(train_label_mix0); % convert to single precision (float 32)
train_y_mix0 = single(train_y_mix0); % convert to single precision (float 32)
save('train_data_mix0_tensorflow_plus.mat','train_data_mix0','train_label_mix0','train_y_mix0');

%parfor i =1 : numel(dev_data_mix0)
for i =1 : numel(dev_data_mix0)
    dev_data_mix0{i} = transpose(padarray(dev_data_mix0{i},[0, max_length - size(dev_data_mix0{i},2)],'post'));
end
tmp = zeros(numel(dev_data_mix0),size(dev_data_mix0{1},1), size(dev_data_mix0{1},2));
%parfor i =1 : numel(dev_data_mix0)
for i =1 : numel(dev_data_mix0)
    tmp(i,:,:) = dev_data_mix0{i};
end
dev_data_mix0 = tmp;
clear tmp
dev_data_mix0 = single(dev_data_mix0); % convert to single precision (float 32)
dev_label_mix0 = single(dev_label_mix0); % convert to single precision (float 32)
dev_y_mix0 = single(dev_y_mix0); % convert to single precision (float 32).
save('dev_data_mix0_tensorflow_plus.mat','dev_data_mix0','dev_label_mix0','dev_y_mix0');

%parfor i =1 : numel(test_data_mix0)
for i =1 : numel(test_data_mix0)
    test_data_mix0{i} = transpose(padarray(test_data_mix0{i},[0, max_length - size(test_data_mix0{i},2)],'post'));
end
tmp = zeros(numel(test_data_mix0),size(test_data_mix0{1},1), size(test_data_mix0{1},2));
%parfor i =1 : numel(test_data_mix0)
for i =1 : numel(test_data_mix0)
    tmp(i,:,:) = test_data_mix0{i};
end
test_data_mix0 = tmp;
clear tmp
test_data_mix0 = single(test_data_mix0); % convert to single precision (float 32)
test_label_mix0 = single(test_label_mix0); % convert to single precision (float 32)
test_y_mix0 = single(test_y_mix0); % convert to single precision (float 32).
save('test_data_mix0_tensorflow_plus.mat','test_data_mix0','test_label_mix0','test_y_mix0');


% padding zeros
%parfor i =1 : numel(train_data_mix10)
for i =1 : numel(train_data_mix10)
    train_data_mix10{i} = transpose(padarray(train_data_mix10{i},[0, max_length - size(train_data_mix10{i},2)],'post'));
end
tmp = zeros(numel(train_data_mix10),size(train_data_mix10{1},1), size(train_data_mix10{1},2));
%parfor i =1 : numel(train_data_mix10)
for i =1 : numel(train_data_mix10)
    tmp(i,:,:) = train_data_mix10{i};
end
train_data_mix10 = tmp;
clear tmp
train_data_mix10 = single(train_data_mix10); % convert to single precision (float 32)
train_label_mix10 = single(train_label_mix10); % convert to single precision (float 32)
train_y_mix10 = single(train_y_mix10); % convert to single precision (float 32)
save('train_data_mix10_tensorflow_plus.mat','train_data_mix10','train_label_mix10','train_y_mix10');

%parfor i =1 : numel(dev_data_mix10)
for i =1 : numel(dev_data_mix10)
    dev_data_mix10{i} = transpose(padarray(dev_data_mix10{i},[0, max_length - size(dev_data_mix10{i},2)],'post'));
end
tmp = zeros(numel(dev_data_mix10),size(dev_data_mix10{1},1), size(dev_data_mix10{1},2));
parfor i =1 : numel(dev_data_mix10)
    tmp(i,:,:) = dev_data_mix10{i};
end
dev_data_mix10 = tmp;
clear tmp
dev_data_mix10 = single(dev_data_mix10); % convert to single precision (float 32)
dev_label_mix10 = single(dev_label_mix10); % convert to single precision (float 32)
dev_y_mix10 = single(dev_y_mix10); % convert to single precision (float 32).
save('dev_data_mix10_tensorflow_plus.mat','dev_data_mix10','dev_label_mix10','dev_y_mix10');

%parfor i =1 : numel(test_data_mix10)
for i =1 : numel(test_data_mix10)
    test_data_mix10{i} = transpose(padarray(test_data_mix10{i},[0, max_length - size(test_data_mix10{i},2)],'post'));
end
tmp = zeros(numel(test_data_mix10),size(test_data_mix10{1},1), size(test_data_mix10{1},2));
%parfor i =1 : numel(test_data_mix10)
for i =1 : numel(test_data_mix10)
    tmp(i,:,:) = test_data_mix10{i};
end
test_data_mix10 = tmp;
clear tmp
test_data_mix10 = single(test_data_mix10); % convert to single precision (float 32)
test_label_mix10 = single(test_label_mix10); % convert to single precision (float 32)
test_y_mix10 = single(test_y_mix10); % convert to single precision (float 32).
save('test_data_mix10_tensorflow_plus.mat','test_data_mix10','test_label_mix10','test_y_mix10');


% padding zeros
%parfor i =1 : numel(train_data_mix20)
for i =1 : numel(train_data_mix20)
    train_data_mix20{i} = transpose(padarray(train_data_mix20{i},[0, max_length - size(train_data_mix20{i},2)],'post'));
end
tmp = zeros(numel(train_data_mix20),size(train_data_mix20{1},1), size(train_data_mix20{1},2));
%parfor i =1 : numel(train_data_mix20)
for i =1 : numel(train_data_mix20)
    tmp(i,:,:) = train_data_mix20{i};
end
train_data_mix20 = tmp;
clear tmp
train_data_mix20 = single(train_data_mix20); % convert to single precision (float 32)
train_label_mix20 = single(train_label_mix20); % convert to single precision (float 32)
train_y_mix20 = single(train_y_mix20); % convert to single precision (float 32)
save('train_data_mix20_tensorflow_plus.mat','train_data_mix20','train_label_mix20','train_y_mix20');

%parfor i =1 : numel(dev_data_mix20)
for i =1 : numel(dev_data_mix20)
    dev_data_mix20{i} = transpose(padarray(dev_data_mix20{i},[0, max_length - size(dev_data_mix20{i},2)],'post'));
end
tmp = zeros(numel(dev_data_mix20),size(dev_data_mix20{1},1), size(dev_data_mix20{1},2));
%parfor i =1 : numel(dev_data_mix20)
for i =1 : numel(dev_data_mix20)
    tmp(i,:,:) = dev_data_mix20{i};
end
dev_data_mix20 = tmp;
clear tmp
dev_data_mix20 = single(dev_data_mix20); % convert to single precision (float 32)
dev_label_mix20 = single(dev_label_mix20); % convert to single precision (float 32)
dev_y_mix20 = single(dev_y_mix20); % convert to single precision (float 32).
save('dev_data_mix20_tensorflow_plus.mat','dev_data_mix20','dev_label_mix20','dev_y_mix20');

%parfor i =1 : numel(test_data_mix20)
for i =1 : numel(test_data_mix20)
    test_data_mix20{i} = transpose(padarray(test_data_mix20{i},[0, max_length - size(test_data_mix20{i},2)],'post'));
end
tmp = zeros(numel(test_data_mix20),size(test_data_mix20{1},1), size(test_data_mix20{1},2));
parfor i =1 : numel(test_data_mix20)
for i =1 : numel(test_data_mix20)    
    tmp(i,:,:) = test_data_mix20{i};
end
test_data_mix20 = tmp;
clear tmp
test_data_mix20 = single(test_data_mix20); % convert to single precision (float 32)
test_label_mix20 = single(test_label_mix20); % convert to single precision (float 32)
test_y_mix20 = single(test_y_mix20); % convert to single precision (float 32).
save('test_data_mix20_tensorflow_plus.mat','test_data_mix20','test_label_mix20','test_y_mix20');