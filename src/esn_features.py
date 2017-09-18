import numpy as np
from numpy import atleast_2d
from sklearn.linear_model import Ridge
from multiprocessing import Pool
import scipy.io as sio
from os.path import join
import os
from simple_esn import  SimpleESN
import re
import pickle
import sys
import time
from sklearn.metrics import mean_squared_error
#np.seterr(all='raise')
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class Data_Set:
    def __init__(self, path,
                 sensor_number=0,
                 name_acq='Measures',
                 size_subsignals=120,
                 overlapping=96,
                 delay=1,
                 reservoir_size=10):
        self.name_acq = name_acq
        self.sensor_number = sensor_number
        self.size_subsignals = size_subsignals
        self.overlapping = overlapping
        self.delay = delay
        self.reservoir_size = reservoir_size
        self.esn = SimpleESN(n_readout=reservoir_size, n_components=reservoir_size, weight_scaling=1.25, damping=0.3)
        self.features, self.labels = self.processing_dataset(path)

    def init_esn(self):
        X = atleast_2d(np.zeros((self.size_subsignals,))).T
        self.esn.fit_transform(X)

    def processing_dataset(self, path):
        files = []
        labels = []
        #self.init_esn()

        for dir in os.listdir(path):
            folder = join(path, dir)
            list_files = [join(folder, file) for file in os.listdir(folder)]
            n_processes = len(list_files)
            labels.extend(os.listdir(folder))
            files.extend(list_files)

        features = []
        labels = []

        #with Pool(processes=45) as p:
        #    feature, label= zip(*p.map(self.feature_extraction_file, files))
        #    features.append(feature)
        #    labels.append(label)
        print('Processes:', n_processes)
        p = Pool(processes=n_processes)
        iterador = p.imap(self.feature_extraction_file, files)
        for iteracion in iterador:
            feature, label = iteracion
            features.append(feature)
            labels.append(label)

        print('Finish')
        p.close()
        return np.array(features).squeeze(), np.array(labels)

    def feature_extraction_file(self, file):
        print(file)
        label = re.split('R|F|L|P|.mat', file)[-5:-1]
        data = sio.loadmat(file)
        signal = data['data'][self.name_acq][0][0][self.sensor_number][:243]
        signal = (signal - signal.min())/(signal.max()-signal.min())
        splitted_signal = rolling_window(signal, self.size_subsignals)

        features = []
        n = 1

        for i in range(self.delay, len(signal)-self.size_subsignals, self.size_subsignals - self.overlapping):
            esn = SimpleESN(n_readout=self.reservoir_size, n_components=self.reservoir_size, weight_scaling=1.25,
                            damping=0.3)
            u = atleast_2d(splitted_signal[i-self.delay,:]).T
            y = atleast_2d(splitted_signal[i,:]).T
            x = esn.transform(u)
            regr = Ridge(alpha=0.01)
            regr.fit(x,y)
            #y_p = regr.predict(x)
            #print(mean_squared_error(y,y_p))
            features.append(np.squeeze(np.concatenate((regr.coef_, atleast_2d(regr.intercept_)), axis=1)))
            #print("n:",n)
            #n = n + 1
        return np.array(features).squeeze(), label


def main(path):
    from esn_features import Data_Set
    dataset = Data_Set(path)
    with open('../data/DB_005V0_l120_10.pickle', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main('/home/titan/Dropbox/mechanical_datasets/DB_005V0/Raw_Data_DB_005V0')
    #main('/home/titan/Dropbox/mechanical_datasets/DB_001V0_test/Raw_Data_DB_001V0')
'''
def processing_signal(signal, label, size_subsignals, overlapping, delay):
    subsignal1, subsignal2 = rolling_window(signal, size_subsignals)

    batch_size = len(subsignal1)
    batch = np.zeros((batch_size, esn.n_readout + 1))
    labels_batch = np.zeros((batch_size, len(labels[idx])))

    for i in range(batch_size):
        activations = esn.transform(subsignal1[i].reshape((-1, 1)))
        regr = Ridge(alpha=0.01)
        regr.fit(activations, subsignal2[i].reshape((-1, 1)))
        batch[i, :] = np.concatenate((regr.coef_, atleast_2d(regr.intercept_)), axis=1).reshape((-1,))
        labels_batch[i, :] = labels[idx]

def feature_extraction_cpu(esn, signals, labels, size_subsignals, overlapping, delay):
    batches = []
    labels_batches = []
    for idx, signal in enumerate(signals):


        batches.append(batch)
        labels_batches.append(labels_batch)
    return batches, labels_batches

def split_signal(signal, size_subsignals, overlapping, delay):
    subsignal1 = []
    subsignal2 = []
    for i in range(0,len(signal), size_subsignals - overlapping):
        if i+size_subsignals+delay <= len(signal):
            subsignal1.append(signal[i:i+size_subsignals])
            subsignal2.append(signal[i+delay:i+size_subsignals+delay])
    return subsignal1, subsignal2

def feature_extraction(sess, esn, data, target, signals, size_subsignals, overlapping, delay):
    batches = []
    for signal in signals:
        subsignal1,subsignal2 = split_signal(signal, size_subsignals, overlapping, delay)
        batch_size  = len(subsignal1)
        batch = np.zeros((batch_size, esn._reservoir_size))
        for i in range(batch_size):
            sess.run(esn.optimize, {data: subsignal1[i].reshape((-1,1)),
                                    target: subsignal2[i].reshape((-1,1))})

            code = sess.run(esn.codification)
            batch[i,:] = code.reshape((-1,))
        batches.append(batch)
    return batches

'''