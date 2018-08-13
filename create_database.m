clear all
close all
clc

% load config
config;

rand('state',0); 

% create data folder if not exist
if(~exist(wave_dir,'dir'))
    mkdir(wave_dir);
end
for i = 1 : numel(wave_dir_mix)
    if(~exist(wave_dir_mix{i},'dir'))
        mkdir(wave_dir_mix{i});
    end
end

% copy clean signal
for cl = 1 : numel(class_name)
    if(~exist([wave_dir, '/', class_name{cl}], 'dir'))
        mkdir([wave_dir, '/', class_name{cl}]);
    end
    disp([wave_dir, '/', class_name{cl}]);
    files = dir(original_wave_dir{cl});
    permlist= randperm(numel(files)-2) + 2;
    for i = 1 : nfile
        if(~exist([wave_dir, '/', class_name{cl}, '/', num2str(i),'.wav'],'file'))
            audiofile = [original_wave_dir{cl},'/',files(permlist(i)).name];
            fid=fopen(audiofile,'r');
            % read the raw audio file
            d=fread(fid,inf,'short');
            fclose(fid);

            % write wave file
            wavefile = [wave_dir, '/', class_name{cl},'/', num2str(i),'.wav'];
            wavwrite(int16(d),Fs,16,wavefile);
        end
    end
end

% create mix signals
for cl = 1 : numel(class_name)
    for i = 1 : nfile
        % read clean signals
        d = wavread([wave_dir,'/', class_name{cl}, '/', num2str(i),'.wav']);
        % randomly select noise file
        noisefile = noise_path{randi([1,numel(noise_path)])};
        for db = 1 : numel(SNR)
            if(~exist([wave_dir_mix{db}, '/', class_name{cl}], 'dir'))
                mkdir([wave_dir_mix{db}, '/', class_name{cl}]);
            end
            disp([wave_dir_mix{db}, '/', class_name{cl}]);
            if(~exist([wave_dir_mix{db}, '/', class_name{cl}, '/', num2str(i),'.wav'],'file'))
                % add noise with random position
                [d_noise, noise] = add_noisem_rand_position(d, noisefile, SNR(db), Fs);
                % write wave file
                wavefile = [wave_dir_mix{db},'/', class_name{cl}, '/', num2str(i),'.wav'];
                wavwrite(d_noise,Fs,wavefile);
            end
        end
    end
end
