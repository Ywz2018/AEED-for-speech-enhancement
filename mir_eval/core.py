# -*- coding: utf-8 -*-
"""
@author: PengChuan
这一部分是语音信号的评价指标，用来评估语音信号降噪的质量，判断结果好坏
    pesq：perceptual evaluation of speech quality，PESQ是语音的听觉质量、听感，范围为-0.5到4.5，也是越高越好
    stoi：short time objective intelligibility，短时客观可懂度，尤其在低SNR下，可懂度尤其重要
        STOI指语音的可懂度，得分在0到1之间，越高越好
    ssnr: segmental SNR，分段信噪比(时域指标)，它是参考信号和信号差的比值，衡量的是降噪程度
"""

import os
import tempfile  # 临时文件夹
import numpy as np
# import librosa  # 用于音频特征提取

import sound.core as sdcore
import util

M_PESQ = "PESQ" # metrics PESQ
M_STOI = "STOI" # metrics STOI
M_SSNR = "SSNR" # metrics SSNR

# 导入pesq.exe，用于语音质量的评估
PESQ_PATH = os.path.split(os.path.realpath(__file__))[0]
if util.os_linux():
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.ubuntu16.exe')
else:
    PESQ_PATH = os.path.join(PESQ_PATH, 'pesq.win10.exe')

# Machine limits for integer types
# Maximum value of given dtype
max_int = np.iinfo(np.int16).max

# (float) The smallest positive usable number.
# Type of tiny is an appropriate floating point type.
min_pf = np.finfo(np.float32).tiny

# 根据人耳对信噪比有意义的范围，做了对过高或过低信噪比的限制
ssnr_min = -40
ssnr_max = 40


def calc_pesq(ref_sig, sig, samplerate, is_file=True):
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


def calc_ssnr(ref_sig, sig, frame_size, mid_only=False):
    '''
    计算分段信噪比
    :param ref_sig: 作为参考（reference）的干净的信号
    :param sig: 待评估的语音信号
    :param frame_size: 帧的size
    :param mid_only:判断是否是中点
    :return: 计算得到的分段信噪比
    '''
    ref_frame = sdcore.frame(ref_sig, frame_size, frame_size)  # 用于生成相应的frame array
    deg_frame = sdcore.frame(sig, frame_size, frame_size)
    if mid_only:  # 如果是从中点的话，只需要计算帧的一半即可，防止计算冗余
        i = len(ref_frame) // 2
        ref_frame = ref_frame[i, :]
        deg_frame = deg_frame[i, :]
    noise_frame = ref_frame - deg_frame
    ref_energy = np.sum(ref_frame ** 2, axis=-1) + min_pf
    noise_energy = np.sum(noise_frame ** 2, axis=-1) + min_pf
    ssnr = 10 * np.log10(ref_energy / noise_energy)
    if mid_only:
        # return min(ssnr_max, max(ssnr_min, ssnr))
        return ssnr
    else:
        ssnr[ssnr < ssnr_min] = ssnr_min
        ssnr[ssnr > ssnr_max] = ssnr_max
        return np.mean(ssnr)
