import dataset_generator as dg
import pickle
from simple_esn import  SimpleESN
from esn_features import feature_extraction_cpu

signals, labels = dg.import_vibration_signal('/home/titan/Dropbox/Chile/acc_signals.mat')

reservoir_size = 1000

esn = SimpleESN(n_readout=reservoir_size, n_components=reservoir_size, weight_scaling=1.25, damping=0.3)

dataset = feature_extraction_cpu(esn=esn,
                                 signals=signals,
                                 labels=labels,
                                 size_subsignals=50000,
                                 overlapping=40000,
                                 delay=1)

with open('data/bignormal1001cpu.pickle', 'wb') as f:
    pickle.dump(batches, f, pickle.HIGHEST_PROTOCOL)