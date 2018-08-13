import numpy as np
import re
import itertools
from collections import Counter

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    #num_batches_per_epoch = int(len(data)/batch_size) + 1
    num_batches_per_epoch = int(len(data)/batch_size)
    if num_batches_per_epoch*batch_size < len(data):
        num_batches_per_epoch = num_batches_per_epoch + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
