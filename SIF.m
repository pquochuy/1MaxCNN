% create spectrogram SIF features 
% wave: audio waveform
% fs: sampling frequency
% ny: number of frequency subbands for downsampling
% winlen: short time window length
% overlap: overlapping length
function data = SIF(wave, fs, ny, winlen, overlap)
    nfft = 2^nextpow2(winlen);
    data0=abs(spectrogram(wave,hamming(winlen),overlap,nfft,fs));
    nchannel = size(data0,1);
    for y = 1 : ny % frequency downsample to keep ny subbands
        data(y,:) = mean(data0(ceil(nchannel/ny*(y-1))+1:ceil(nchannel/ny*y),:));
    end
    for y = 1 : ny % denoising
        data(y,:)=data(y,:) - min(data(y,:));
    end;
end