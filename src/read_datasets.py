import pickle
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
from sklearn.model_selection import StratifiedKFold
import copy


class DataSet(object):
    def __init__(self,
                 data,
                 labels,
                 patterns,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype != dtypes.float32:
            raise TypeError('Invalid data dtype %r, expected float32' % dtype)

        assert data.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (data.shape, labels.shape))

        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._patterns = patterns

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def patterns(self):
        return self._patterns

    @property
    def num_examples(self):
        return self._num_examples

    def data_by_condition(self, condition):
        return self._data[(self._labels == condition).all(axis=1)]

    def num_examples_by_condition(self, condition):
        return np.count_nonzero((self._labels == condition).all(axis=1))

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

    def batch_test(self, condition):
        print(self._data.shape)
        return self._data[(self._labels == condition).all(axis=1)], \
               self._labels[(self._labels == condition).all(axis=1)]

def format_data_labels(data, labels):
    data = data.astype(np.float32)
    labels = labels.astype(np.int8)
    labels = np.repeat(labels,data.shape[1],axis=0)
    return data.reshape((data.shape[0]*data.shape[1],-1)), labels


def read_datasets(path_data):
    with open(path_data, 'rb') as f:
        dataset = pickle.load(f)
    sss = StratifiedKFold(n_splits=2, random_state=0)
    labels = dataset.labels.astype(np.int8)
    labels = labels.squeeze()
    print(labels)
    features_filtered = dataset.features[labels[:,1] != 4]
    labels_filtered = labels[labels[:,1] != 4]
    features_filtered = features_filtered[labels_filtered[:, 1] != 5]
    labels_filtered = labels_filtered[labels_filtered[:, 1] != 5]
    mask = [str(lab[0])+str(lab[1])+str(lab[2]) for lab in labels_filtered[labels_filtered[:, 3] == 1, 1:]]

    for train_index, test_index in sss.split(np.zeros(len(mask)), mask):
        data_one_train = features_filtered[train_index,:,:]
        data_one_test =  features_filtered[test_index,:,:]
        labels_one_train = labels_filtered[train_index, :]
        labels_one_test = labels_filtered[test_index, :]
        break
    print(labels_one_test.shape)
    data_train, labels_train = format_data_labels(data_one_train, labels_one_train)
    data_test, labels_test = format_data_labels(data_one_test, labels_one_test)

    data_two_test, labels_two_test = format_data_labels(features_filtered[labels_filtered[:, 3] != 1,:],
                                                        labels_filtered[labels_filtered[:, 3] != 1,:])
    print(labels_one_test.shape)
    patterns_test = np.vstack((labels_one_test, labels_filtered[labels_filtered[:, 3] != 1,:]))
    patterns_eval = copy.copy(patterns_test)
    data_test = np.vstack((data_test,data_two_test))
    print(labels_test)
    labels_test = np.vstack((labels_test,labels_two_test))
    data_valid = copy.copy(data_test)
    labels_valid = copy.copy(labels_test)

    train = DataSet(data_train, labels_train, labels_one_train)
    validation = DataSet(data_valid, labels_valid, patterns_eval)
    test = DataSet(data_test, labels_test, patterns_test)

    return base.Datasets(train=train, validation=validation, test=test)

def read_datasets2(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    train, test = data
    valid = copy.copy(test)
    data_train, labels_train = train
    data_test, labels_test = test
    data_valid, labels_valid = valid

    data_train = np.squeeze(data_train)
    data_test = np.squeeze(data_test)
    data_valid = np.squeeze(data_valid)

    data_train = data_train[labels_train[:, 3] == 1]
    labels_train = labels_train[labels_train[:, 3] == 1]

    data_train = data_train[labels_train[:, 1] != 4]
    labels_train = labels_train[labels_train[:, 1] != 4]

    data_train = data_train[labels_train[:, 1] != 5]
    labels_train = labels_train[labels_train[:, 1] != 5]

    print('Forma:',data_train.shape)
    train2 = DataSet(data_train, labels_train,labels_train)
    validation2 = DataSet(data_valid, labels_valid,labels_valid)
    test2 = DataSet(data_test, labels_test,labels_test)
    return base.Datasets(train=train2, validation=validation2, test=test2)