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


class Data_Set_Ts:
    def __init__(self, path,
                 sensor_number=0,
                 name_acq='Measures',
                 size_subsignals=1000,
                 overlapping=0):
        self.name_acq = name_acq
        self.sensor_number = sensor_number
        self.size_subsignals = size_subsignals
        self.overlapping = overlapping
        self.features, self.labels = self.processing_dataset(path)

    def processing_dataset(self, path):
        files = []
        labels = []

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
        signal = data['data'][self.name_acq][0][0][self.sensor_number][:5000]
        signal = (signal - signal.min())/(signal.max()-signal.min())
        splitted_signal = rolling_window(signal, self.size_subsignals)
        features = []

        for i in range(0, len(signal)-self.size_subsignals, self.size_subsignals - self.overlapping):
            features.append(splitted_signal[i,:])

        return np.array(features).squeeze(), label


def main(path):
    from ts_features import Data_Set_Ts
    dataset = Data_Set_Ts(path)
    with open('../data/DB_005V0_ts1000_0.pickle', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main('/home/titan/Dropbox/mechanical_datasets/DB_005V0/Raw_Data_DB_005V0')
    #main('/home/titan/Dropbox/mechanical_datasets/DB_001V0_test/Raw_Data_DB_001V0')