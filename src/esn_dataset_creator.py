import sys
sys.path.append("../Echo_State_Networks")
import tensorflow as tf
import dataset_generator as dg
import esn_features as esn_features
import pickle
from ESNs.standard import EsnNetwork #Crea una ESN

sess = tf.Session()

normal_signals, anomalous_signals = dg.import_vibration_signal('/home/dcabrera/Dropbox/Chile/acc_signals.mat')
print(normal_signals.shape)
print(anomalous_signals.shape)

prec = tf.float64
reservoir_size = 1000
data = tf.placeholder(prec, [None, 1])
target = tf.placeholder(prec, [None, 1])
esn = EsnNetwork(data, target, reservoir_size=reservoir_size, prec=prec)
sess.run(tf.global_variables_initializer())

batches = esn_features.feature_extraction(sess=sess, esn=esn, data=data,
                                          target=target,
                                          signals=normal_signals,
                                          size_subsignals=12000,
                                          overlapping=100,delay=1)

with open('data/normal1000.pickle', 'wb') as f:
    pickle.dump(batches, f, pickle.HIGHEST_PROTOCOL)

batches = esn_features.feature_extraction(sess=sess, esn=esn, data=data,
                                          target=target,
                                          signals=anomalous_signals,
                                          size_subsignals=12000,
                                          overlapping=100,delay=1)
with open('data/anomalous1000.pickle', 'wb') as f:
    pickle.dump(batches, f, pickle.HIGHEST_PROTOCOL)

sess.close()