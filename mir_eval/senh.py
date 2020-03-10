# -*- coding: utf-8 -*-
"""
@author: PengChuan
针对语音增强的评价模块
"""
import os
import librosa
from mir_eval.core import calc_pesq, calc_ssnr
from assess.stoi import calc_stoi


def eval_dir(enh_root, cln_root=None, msr_list=('stoi', 'pesq', 'ssnr'), sr=16000, align=False):
    '''
    对指定目录里的所有语音做评价，包括stoi, pesq, ssnr等。干净语音所在目录中，
    包含干净语音"干净语音.wav"，在增强语音所在目录中，包含对应增强语音“干净语音+噪声-sep.wav”。
    :param enh_root: 增强语音所在目录
    :param cln_root: 干净语音所在目录，如果None，默认与enh_root相同
    :param msr_list: 评价指标
    :param sr: 采样频率
    :param align: 当增强语音与干净语音不等长时，是否剪去多余部分使之等长
    :return:
    '''
    if cln_root is None:
        cln_root = enh_root

    files = os.listdir(enh_root)

    score = [0. for _ in range(len(msr_list))]
    count = 0
    for file in files:
        if not file.endswith('-sep.wav'):
            continue

        enh, _ = librosa.load(os.path.join(enh_root, file), sr=sr)
        cln_name = file[:file.index('+')] + '.wav'
        cln, _ = librosa.load(os.path.join(cln_root, cln_name), sr=sr)

        if align:
            if len(enh) < len(cln):
                cln = cln[:len(enh)]
            elif len(enh) > len(cln):
                enh = enh[:len(cln)]

        for i in range(len(msr_list)):
            if msr_list[i] == 'stoi':
                score[i] += calc_stoi(cln, enh, sr)
            elif msr_list[i] == 'pesq':
                score[i] += calc_pesq(cln, enh, sr, is_file=False)
            elif msr_list[i] == 'ssnr':
                score[i] += calc_ssnr(cln, enh, int(sr * 0.02))
        count += 1

    return [s / count for s in score]


if __name__ == '__main__':
    import sys

    enh_root = sys.argv[1]
    cln_root = sys.argv[2]
    msr_list = sys.argv[3:]

    score = eval_dir(enh_root, cln_root, msr_list, align=True)
    print(score)
