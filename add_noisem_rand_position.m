function [Y,NOISE] = add_noisem_rand_position(X, filepath_name, SNR, fs)
% add_noisem add determinated noise to a signal.
% X is signal, and its sample frequency is fs;
% filepath_name is NOISE's path and name, and the SNR is signal to noise ratio in dB.
[wavin,fs1,nbits]=wavread(filepath_name);
nx=size(X,1);

if fs1~=fs
    wavin=resample(wavin,fs,fs1);
end
start_pos = randi([1, (length(wavin) - nx + 1)]);
NOISE=wavin(start_pos : (start_pos + nx - 1));


NOISE=NOISE-mean(NOISE);
signal_power = 1/nx*sum(X.*X);
noise_variance = signal_power / ( 10^(SNR/10) );
NOISE=sqrt(noise_variance)/std(NOISE)*NOISE;
Y=X+NOISE;