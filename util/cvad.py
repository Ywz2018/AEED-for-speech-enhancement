# -*- coding: utf-8 -*-

"""
对干净（clean，不含噪声）语音音频的话音检测（vad），用第三方库webrtcvad实现。
webrtcvad采用基于能量的检测方法，标记语音分片，但不适用于含有噪声的情况。
代码修改自官方github库的example.py：
https://github.com/wiseman/py-webrtcvad
"""

import collections
import contextlib
import wave
import webrtcvad


def _read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def _write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class _Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def _frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield _Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def _vad_collector(sample_rate, frame_duration_ms,
                   padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        # sys.stdout.write(
        #     '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
    #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def cutoff_muse(origin, out_path, aggressive=3, duration=30, pad_duration=300):
    '''
    用webrtcvad第三方库剪切音频中的静音部分，输出到另一个音频中
    :param origin:  原始音频，采样位数必须为16，采样频率8kHz或16kHz
    :param out_path:    输出音频路径
    :param aggressive:  检测强度，0至3，int类型，0最弱倾向保留音频
    :param duration:    每帧检测时长，单位ms
    :param pad_duration:    检测语音端点的缓冲区大小，单位ms
    :return:
    '''
    audio, sample_rate = _read_wave(origin)
    vad = webrtcvad.Vad(aggressive)
    frames = _frame_generator(duration, audio, sample_rate)
    frames = list(frames)

    segments = _vad_collector(sample_rate, duration, pad_duration, vad, frames)
    segments = list(segments)
    voice = b''.join(segments)

    _write_wave(out_path, voice, sample_rate)
