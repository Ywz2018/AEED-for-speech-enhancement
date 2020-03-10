import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import pickle


def load_train_record(filename):
    file = open(filename, 'rb')
    record = []
    while True:
        try:
            record.append(pickle.load(file))
        except EOFError:
            return record


def plot_sample_dist(sub_plt=False):
    record = load_train_record('train_record_ori2.pkl')
    x = []
    score = []
    for one in record:
        for sample in one:
            two = 2
            ssnr, count = 0., 0
            for r in sample:
                if two >= 1:
                    two -= 1
                    continue
                if type(r) == list:
                    ssnr += r[0]
                    count += 1
                    x.append(r)
                else:
                    score.append([ssnr / count, r])
            del x[-1]
            del x[-1]
    # x = [[0,1,2,3], []  ]  0 - SSNR; 1 - true logit; 2 - fake logit; 3 - MSE

    x = np.array(x)
    length = len(x) // 100
    print('record shape:')
    print(x.shape)

    i = 1
    _x = x[i * length:(i + 1) * length]

    print('sample num:')
    print(_x.shape[0])
    print('min SSNR:')
    print(np.min(x[:, 0]))

    if not sub_plt:
        plt.figure('D_loss')
    plt.plot(_x[:, 1], _x[:, 0], 'r.', label='Clean Sample')
    plt.plot(_x[:, 2], _x[:, 0], 'b.', label='Enhancement Sample')
    plt.ylabel('SSNR')
    plt.xlabel('Logits')
    plt.legend(loc='best')
    if not sub_plt:
        plt.show()
        plt.savefig("SSNR-D_loss.png")


def plot_wav():
    y, sr = librosa.load('./dataset/Test/0/SA1_DR1+hfchannel.wav', sr=16000)
    window = 640
    plt.axis([0, window, 0, 1])
    plt.ion()

    start = 0
    for i in range(window):
        # plt.scatter(i, y)
        plt.cla()
        plt.axis([0, window, -1, 1])
        plt.plot(np.arange(window), y[start:start + window])
        plt.pause(0.1)
        start += int(window / 2)


def plot_spec(wav_path, sub_plt=False):
    s, fs = librosa.load(wav_path, sr=None)

    if not sub_plt:
        plt.figure('spec')
    plt.specgram(s, Fs=16000, scale_by_freq=True, sides='default')
    plt.ylabel('Frequency(Hz)')
    plt.xlabel('Time(s)')

    if not sub_plt:
        plt.show()


def plot_all():
    plt.figure(1, figsize=(100, 100))
    plt.subplot(311)
    plot_sample_dist(sub_plt=True)

    plt.subplot(334)
    plot_spec('dataset/Test/0/SA1_DR1+hfchannel.wav', sub_plt=True)
    plt.subplot(335)
    plot_spec('dataset/Test/0/SA1_DR1+hfchannel-sep.wav', sub_plt=True)
    plt.subplot(336)
    plot_spec('dataset/Test/0/SA1_DR1.wav', sub_plt=True)

    plt.subplot(313)
    plot_wav()
