This experiment setup for the published paper:
Huy Phan, Lars Hertel, Marco Maass, and Alfred Mertins. **Audio Event Recognition with 1-Max Pooling Convolutional Neural Networks.** _In Proceedings of 17th Annual Conference of the International Speech Communication Association (INTERSPEECH)_, pp. 3653-3657, 2016

The setup is similar to that in 

I. McLoughlin, H.-M. Zhang, Z.-P. Xie, Y. Song, W. Xiao, **Robust Sound Event Classification using Deep Neural Networks**, _IEEE Trans. Audio, Speech and Language Processing_, Jan 2015

(which can be found here http://www.lintech.org/machine_hearing/index.html)

The implementation of 1-max pooling CNN is based on the implementation of Denny Britz:

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/


Although implementation with variable length is possible, we padded zeros each input to the maximum length to ease the implementation.These zeros paddings actually do not bring significant affects on the 1-max pooling.

The experiments can be run as follows:

**On MATLAB:** (for SIF feature extraction)

create_database;  
extract_sif;  
extract_sif_plus;  
export_data_tensorflow;  
export_data_tensorflow_plus;  
export_data_multicondition_tensorflow;  
export_data_multicondition_tensorflow_plus;  

**On Tensorflow:** (for CNN training and evaluation)

bash run.sh

