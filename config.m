noisex92_source = './noisex-92/';
RWCP_SSD_source = '~/Dataset/RWCP-SSD/RWCP-SSD_vol1/nospeech/drysrc/';

noise_path = {  [noisex92_source, 'babble.wav'];
                            [noisex92_source, 'factory1.wav'];
                            [noisex92_source, 'f16.wav'];
                            [noisex92_source, 'destroyerengine.wav']};
wave_dir = './data_wav';
SNR = [0,10,20];
wave_dir_mix = {'./data_wav_mix0';
                            './data_wav_mix10';
                            './data_wav_mix20'};

original_wave_dir = {   [RWCP_SSD_source,'b4/aircap/16khz'];
                                       [RWCP_SSD_source,'a2/bank/16khz'];
                                       [RWCP_SSD_source,'c1/bells5/16khz'];
                                       [RWCP_SSD_source,'a4/bottle1/16khz'];
                                       [RWCP_SSD_source,'a4/bottle2/16khz'];
                                       [RWCP_SSD_source,'a2/bowl/16khz'];
                                       [RWCP_SSD_source,'c4/buzzer/16khz'];
                                       [RWCP_SSD_source,'a2/candybwl/16khz'];
                                       [RWCP_SSD_source,'b5/cap1/16khz'];
                                       [RWCP_SSD_source,'a3/case1/16khz'];
                                       [RWCP_SSD_source,'a3/case3/16khz'];
                                       [RWCP_SSD_source,'c3/castanet/16khz'];
                                       [RWCP_SSD_source,'c4/clock2/16khz'];
                                       [RWCP_SSD_source,'a2/coffcan/16khz'];
                                       [RWCP_SSD_source,'c1/coin1/16khz'];
                                       [RWCP_SSD_source,'c1/coins1/16khz'];
                                       [RWCP_SSD_source,'a2/colacan/16khz'];
                                       [RWCP_SSD_source,'a3/dice1/16khz'];
                                       [RWCP_SSD_source,'c5/doorlock/16khz'];
                                       [RWCP_SSD_source,'c5/dryer/16khz'];
                                       [RWCP_SSD_source,'b3/file/16khz'];
                                       [RWCP_SSD_source,'c3/horn/16khz'];
                                       [RWCP_SSD_source,'c3/kara/16khz'];
                                       [RWCP_SSD_source,'c3/maracas/16khz'];
                                       [RWCP_SSD_source,'c5/mechbell/16khz'];
                                       [RWCP_SSD_source,'a2/metal10/16khz'];
                                       [RWCP_SSD_source,'a2/metal15/16khz'];
                                       [RWCP_SSD_source,'c5/padlock/16khz'];
                                       [RWCP_SSD_source,'a2/pan/16khz'];
                                       [RWCP_SSD_source,'b1/particl1/16khz'];
                                       [RWCP_SSD_source,'b1/particl2/16khz'];
                                       [RWCP_SSD_source,'c4/phone4/16khz'];
                                       [RWCP_SSD_source,'c4/pipong/16khz'];
                                       [RWCP_SSD_source,'b2/pump/16khz'];
                                       [RWCP_SSD_source,'c5/punch/16khz'];
                                       [RWCP_SSD_source,'c3/ring/16khz'];
                                       [RWCP_SSD_source,'b3/sandpp1/16khz'];
                                       [RWCP_SSD_source,'b3/sandpp2/16khz'];
                                       [RWCP_SSD_source,'c5/shaver/16khz'];
                                       [RWCP_SSD_source,'b5/snap/16khz'];
                                       [RWCP_SSD_source,'b2/spray/16khz'];
                                       [RWCP_SSD_source,'c5/stapler/16khz'];
                                       [RWCP_SSD_source,'b4/sticks/16khz'];
                                       [RWCP_SSD_source,'c3/string/16khz'];
                                       [RWCP_SSD_source,'c2/tear/16khz'];
                                       [RWCP_SSD_source,'c5/toy/16khz'];
                                       [RWCP_SSD_source,'a2/trashbox/16khz'];
                                       [RWCP_SSD_source,'c3/whistle1/16khz'];
                                       [RWCP_SSD_source,'c3/whistle2/16khz'];
                                       [RWCP_SSD_source,'c3/whistle3/16khz']
                                       };
                                       
class_name = {'aircap';
                            'bank';
                            'bells5';
                            'bottle1';
                            'bottle2';
                            'bowl';
                            'buzzer';
                            'candybwl';
                            'cap1';
                            'case1';
                            'case3';
                            'castanet';
                            'clock2';
                            'coffcan';
                            'coin1';
                            'coins1';
                            'colacan';
                            'dice1';
                            'doorlock';
                            'dryer';
                            'file';
                            'horn';
                            'kara';
                            'maracas';
                            'mechbell';
                            'metal10';
                            'metal15';
                            'padlock';
                            'pan';
                            'particl1';
                            'particl2';
                            'phone4';
                            'pipong';
                            'pump';
                            'punch';
                            'ring';
                            'sandpp1';
                            'sandpp2';
                            'shaver';
                            'snap';
                            'spray';
                            'stapler';
                            'sticks';
                            'string';
                            'tear';
                            'toy';
                            'trashbox';
                            'whistle1';
                            'whistle2';
                            'whistle3';
};
                        
Fs = 16000;
nclass=50;
nfile=80;
ny=52;
winlen=1600; % 100 ms
overlap=winlen-160; % advance 10 ms