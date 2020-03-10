'''
Function calc_stoi() calculates the output of the short-time objective intelligibility (STOI) 
measure described in [1, 2].

References:
[1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
    Objective Intelligibility Measure for Time-Frequency Weighted Noisy
    Speech', ICASSP 2010, Texas, Dallas.

[2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for 
    Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', 
    IEEE Transactions on Audio, Speech, and Language Processing, 2011.

modified from Matlab code:
Copyright 2009: Delft University of Technology, Signal & Information
Processing Lab. The software is free for non-commercial use. This program
comes WITHOUT ANY WARRANTY.
    目前知道有stoi这个评价指标就可以了，可以不细看
    STOI指语音的可懂度，得分在0到1之间，越高越好
'''

import numpy as np
# import librosa
import util
import tempfile  # 临时文件夹
import os

def _stdft(x, N, K, N_fft):
    frames = np.arange(0, len(x) - N, K, dtype=np.int32)
    x_stdft = []
    w = _sym_hanning(N)

    for i in range(len(frames)):
        frame = x[frames[i]:frames[i] + N]
        x_stdft.append(np.fft.fft(frame * w, N_fft))

    return np.array(x_stdft)


def _rm_silent_frame(x, y, rng, N, K):
    x, y = np.array(x), np.array(y)
    frames = [i for i in range(0, len(x) - N, K)]
    w = _sym_hanning(N)
    msk = []

    for i in range(len(frames)):
        f = x[frames[i]:frames[i] + N]
        m = 20 * np.log10(np.linalg.norm(f * w) / np.sqrt(N))
        msk.append(m)

    msk = np.array(msk)
    msk = (msk - np.max(msk) + rng) > 0
    count = 0

    x_sil = [0 for i in range(len(x))]
    y_sil = [0 for i in range(len(y))]
    jj_o_stop = 0
    for j in range(len(frames)):
        if msk[j]:
            jj_i, jj_i_stop = frames[j], frames[j] + N
            jj_o, jj_o_stop = frames[count], frames[count] + N
            x_sil[jj_o:jj_o_stop] = x_sil[jj_o:jj_o_stop] + x[jj_i:jj_i_stop] * w
            y_sil[jj_o:jj_o_stop] = y_sil[jj_o:jj_o_stop] + y[jj_i:jj_i_stop] * w
            count += 1

    return x_sil[:jj_o_stop], y_sil[:jj_o_stop]


def _sym_hanning(n):
    def calc_hanning(m, n):
        hann = []
        for i in range(1, m + 1):
            hann.append(.5 * (1 - np.cos(2 * np.pi * i / (n + 1))))
        return hann

    if n % 2:
        half = (n + 1) // 2
        w = calc_hanning(half, n)
        w = w + w[-2::-1]
    else:
        half = n // 2
        w = calc_hanning(half, n)
        w = w + w[::-1]
    return np.array(w)


def _thirdoct(fs, N_fft, numBands, mn):
    step = float(fs) / N_fft
    f = np.array([i * step for i in range(0, N_fft // 2 + 1)])
    k = np.arange(numBands)
    cf = 2 ** (k / 3) * mn
    fl = np.sqrt(cf * 2 ** ((k - 1.) / 3) * mn)
    fr = np.sqrt(cf * 2 ** ((k + 1.) / 3) * mn)
    A = [[0 for j in range(len(f))] for i in range(numBands)]

    for i in range(len(cf)):
        fl_i = np.argmin((f - fl[i]) ** 2)
        fr_i = np.argmin((f - fr[i]) ** 2)
        for j in range(fl_i, fr_i):
            A[i][j] = 1
    A = np.array(A)
    rnk = np.sum(A, axis=1)

    for i in range(len(rnk) - 2, 0, -1):
        if rnk[i + 1] >= rnk[i] and rnk[i + 1] != 0:
            numBands = i + 1
            break

    A = A[0:numBands + 1, :]
    cf = cf[0:numBands + 1]

    return A, cf


def _correlation_coefficient(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    x = x / max(np.sqrt(np.sum(x ** 2)), np.finfo(np.float64).eps)
    y = y / max(np.sqrt(np.sum(y ** 2)), np.finfo(np.float64).eps)
    return np.sum(x * y)


def calc_stoi(clean_sig, bad_sig, fs_signal):
    if len(clean_sig) != len(bad_sig):
        raise ValueError('the length of clean signal and bad signal not equal')

    x, y = np.array(clean_sig), np.array(bad_sig)

    fs = 10000
    N_frame = 256
    K = 512
    J = 15
    mn = 150
    H, _ = _thirdoct(fs, K, J, mn)
    N = 30
    Beta = -15
    dyn_range = 40


    if fs_signal != fs:
        x = util.resample(x, fs_signal, fs)
        y = util.resample(y, fs_signal, fs)

    x, y = _rm_silent_frame(x, y, dyn_range, N_frame, N_frame // 2)
    if len(x) <= 0:
        raise ValueError("Signal contains no speech fragments")

    x_hat = _stdft(x, N_frame, N_frame / 2, K)
    y_hat = _stdft(y, N_frame, N_frame / 2, K)

    # freq*frame
    x_hat = np.transpose(x_hat[:, 0:K // 2 + 1])
    y_hat = np.transpose(y_hat[:, 0:K // 2 + 1])

    X, Y = [], []

    for i in range(x_hat.shape[1]):
        # frame*freq
        X.append(np.sqrt(H.dot(np.abs(x_hat[:, i]) ** 2)))
        Y.append(np.sqrt(H.dot(np.abs(y_hat[:, i]) ** 2)))
    X = np.array(X)
    Y = np.array(Y)
    # freq*frame
    X = X.T
    Y = Y.T

    c = 10 ** (-Beta / 20.)

    score, count = 0., 0
    for m in range(N, X.shape[1] + 1):
        # freq*N
        X_seg = X[:, m - N:m]
        Y_seg = Y[:, m - N:m]

        # freq
        Y_square_sum = np.sum(np.square(Y_seg), axis=1)
        Y_square_sum[Y_square_sum <= 0] = np.finfo(np.float64).eps

        # freq
        alpha = np.sqrt(np.sum(np.square(X_seg), axis=1) / Y_square_sum)
        # freq*1
        alpha = np.reshape(alpha, [len(alpha), 1])
        # freq*N
        aY_seg = Y_seg * np.tile(alpha, [1, N])

        for j in range(J):
            # X_seg[j,:] : N
            # aX : N
            aX = X_seg[j, :] + X_seg[j, :].dot(c)
            Y_prime = [min(x, y) for x, y in zip(aY_seg[j, :], aX)]
            # N
            Y_prime = np.array(Y_prime)
            # 1
            s = _correlation_coefficient(X_seg[j, :], Y_prime)
            score += s
            count += 1

    score /= max(count, 1)

    return score

def calc_pesq(ref_sig, sig, samplerate, is_file=False):
    '''
    计算语音质量听觉评估，调用了当前目录下的pesq.ubuntu16.exe
    return 评估的分数，分数高的结果比较好
    PESQ是语音的听觉质量、听感，范围为-0.5到4.5，分数越高越好
    :param ref_sig: 作为参考（reference）的干净的信号
    :param sig: 待评估的语音信号
    :param samplerate:信号采样率
    :param is_file:是否是文件
    :return:评测的pesq分数
    '''
    PESQ_PATH = os.path.split(os.path.realpath(__file__))[0]

    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.ubuntu16.exe')

    # Machine limits for integer types
    # Maximum value of given dtype
    max_int = np.iinfo(np.int16).max

    # (float) The smallest positive usable number.
    # Type of tiny is an appropriate floating point type.
    min_pf = np.finfo(np.float32).tiny
    if util.os_windows():
        # 暂不支持windows下pesq计算
        return 0

    if is_file:
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, ref_sig, sig))
        msg = output.read()
    else:
        tmp_ref = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_deg = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        util.wav_write(tmp_ref.name, (ref_sig * max_int).astype(np.int16), samplerate)
        util.wav_write(tmp_deg.name, (sig * max_int).astype(np.int16), samplerate)
        output = os.popen('%s +%d %s %s' % (PESQ_PATH, samplerate, tmp_ref.name, tmp_deg.name))
        msg = output.read() # 读取输出的结果字符串
        # print(msg)
        tmp_ref.close()
        tmp_deg.close()
        os.unlink(tmp_ref.name)
        os.unlink(tmp_deg.name)
        # try:
        #     util.rm_r('_pesq_itu_results.txt')
        #     util.rm_r('_pesq_results.txt')
        # except FileNotFoundError:
        #     pass
    try:
        msg_ = msg.split('P.862 Prediction (Raw MOS, MOS-LQO):  = ')
        score = -0.5 if len(msg_) <= 0 else msg_[1].split()[0] # 从结果字符串中截取评分
        return float(score)
    except Exception as err:
        # print(msg)
        # raise err
        return -0.5

def eval_SNR(signal,noisy_signal):
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