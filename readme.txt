This experiment setup for "robust audio event recognition with 1-max pooling convolutional neural networks". The setup is similar to that in 

McLoughlin I, Zhang H.-M., Xie Z.-P., Song Y., Xiao. W., “Robust Sound Event Classification using Deep Neural Networks”, IEEE Trans. Audio, Speech and Language Processing, Jan 2015

(which can be found here http://www.lintech.org/machine_hearing/index.html)

The implementation of 1-max pooling CNN is based on the implementation of Denny Britz:

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/



Although implementation with variable length is possible, we padded zeros each input to the maximum length to ease the implementation.These zeros paddings actually do not bring significant affects on the 1-max pooling.

The experiments can be run as follows:

on MATLAB:

create_database;
extract_sif;
extract_sif_plus;
export_data_tensorflow;
export_data_tensorflow_plus;
export_data_multicondition_tensorflow;
export_data_multicondition_tensorflow_plus;

on tensorflow:

bash run.sh

