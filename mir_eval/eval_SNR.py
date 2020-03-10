import numpy as np
def eval(signal,noisy_signal):
    # signal = np.array()  ## input orignal data
    # mean_signal = np.mean(signal)
    # signal_diff = signal - mean_signal
    # var_signal = np.sum(np.mean(signal_diff ** 2))  ## variance of orignal data
    #
    # # noisy_signal = np.array()  ## input noisy data
    # noise = noisy_signal - signal
    # mean_noise = np.mean(noise)
    # noise_diff = noise - mean_noise
    # var_noise = np.sum(np.mean(noise_diff ** 2))  ## variance of noise

    n_2=np.sum(np.square(noisy_signal-signal))
    s_2=np.sum(np.square(signal))

    if n_2 == 0:
        snr = 100  ## clean image
    else:
        snr = (np.log10(s_2 / n_2)) * 10  ## SNR of the data

    return snr