# -*- coding: utf-8 -*-

"""
各种工具方法，服务于各处理模块
"""

import os
import shutil
import nnresample
import numpy as np
import soundfile
import platform
import xlrd
import logging
import io
from urllib.request import urlopen
import librosa

def mkdir_p(dir_path):
    '''
    mkdri -p，创建目录，不存在则创建，如必要也创建上级目录
    :param dir_path: 目录路径
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def rm_r(path):
    '''
       删除操作
    '''
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def wav_read(path, samplerate=None):
    y, sr = soundfile.read(path)
    #y,sr = soundfile.read(io.BytesIO(urlopen(path).read()))
    if samplerate is not None and samplerate != sr:
        y = resample(y, sr, samplerate)
        sr = samplerate
    return y, sr


def wav_write(path, wav, sr, norm=False):
    if norm:
        wav = wav / np.max(np.abs(wav)) * 0.99999  # 所有的乘以0.9999，<1
    soundfile.write(path, wav, sr)


def resample(wav, old_sr, new_sr):
    return nnresample.resample(wav, new_sr, old_sr)


def os_windows():
    return 'Windows' in platform.system()


def os_linux():
    return 'Linux' in platform.system()


def read_excel(excel_name, sheet_name=None, sheet_index=0, keep_empty=True):
    '''
    读入excel表，获取数据。
    :param excel_name: 表名
    :param sheet_name: sheet名，如果为None，则根据sheet_index读取
    :param sheet_index: sheet索引，在sheet_name为None时有效
    :param keep_empty: 是否保留‘’字段
    :return:
    '''
    bk = xlrd.open_workbook(excel_name)
    if sheet_name is not None:
        sh = bk.sheet_by_name(sheet_name)
    else:
        sh = bk.sheet_by_index(sheet_index)
    nrows = sh.nrows  # 获取行数
    row_list = []
    for i in range(0, nrows):  # 获取各行数据
        row_data = sh.row_values(i)
        if not keep_empty:
            tmp = []
            for c in row_data:
                if c == '':
                    continue
                tmp.append(c)
            row_data = tmp
            if len(row_data) <= 0:
                continue
        row_list.append(row_data)

    return row_list

def copy_file(source,dst):
    '''
    将source文件夹中的文件拷贝到dst文件夹中
    :param source: 源文件所在的文件夹
    :param dst: 目标文件所在的文件夹
    '''
    # source = '/home/pwx/uestc/code/denoising/cache/f54e277d8ff47f4eb918d3d35d0d3de1/test.out/5'
    # destination = '/home/pwx/uestc/code/dataset_clean/f54e/5'
    lst = os.listdir(source)
    for item in lst:
        source_file = os.path.join(source, item)
        shutil.copy2(source_file, dst)

    # for item in lst:
    #     if item.endswith('.enh.wav'):
    #         pass
    #     elif item.endswith('.wav'):
    #         src = os.path.join(source, item)
    #         shutil.copy2(src, dst)


def path_exists(path, log=True):
    if not os.path.exists(path):
        if log:
            currdir = os.path.abspath(os.path.curdir)
            logging.warning('cannot find %s, and current dir is %s' % (path, currdir))
        return False
    return True
