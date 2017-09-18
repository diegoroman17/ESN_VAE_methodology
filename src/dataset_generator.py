import numpy as np
import scipy.io as sio


def sine_signals(f_min=1000, f_max=1100, sampling_f=50000, n_signals=10):
    frequencies = np.linspace(f_min, f_max, n_signals)
    t = np.arange(0, 1, 1 / sampling_f)
    signals = np.zeros((frequencies.shape[0], sampling_f))
    for i,freq in enumerate(frequencies):
        signals[i,:] = np.sin(2 * np.pi * freq * t)
    return signals


def corrupt_sine_signals(f_min=1000, f_max=1100, sampling_f=50000, n_signals=10):
    signals = (0.25 * np.random.random((n_signals, sampling_f))) + \
              sine_signals(f_min=f_min, f_max=f_max, sampling_f=sampling_f, n_signals=n_signals)
    return signals

def import_vibration_signal(path):
    m_data = sio.loadmat(path)
    labels = m_data['param_signals']
    signals = m_data['signals']
    return signals, labels

def main():
    '''
    signals = sine_signals()
    print(signals.shape)
    signals = corrupt_sine_signals()
    print(signals.shape)
    '''
    normal_signals, anomalous_signals = import_vibration_signal('/home/dcabrera/Dropbox/Chile/acc_signals.mat')
    print(normal_signals.shape)
    print(anomalous_signals.shape)


if __name__ == "__main__":
    main()
