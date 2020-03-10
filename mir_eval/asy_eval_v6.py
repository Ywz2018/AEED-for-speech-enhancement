# -*- coding: utf-8 -*-
"""
@author: PengChuan
配合prep_data_v6，run_eval中voice输入变为单元素
"""

from multiprocessing import Process, Queue

from torch.autograd import Variable
import numpy as np
import assess
import util


EPS = np.finfo(float).eps

def run_eval(net, x, y, samplerate, n_proc=7, buffersize=16, raw_mix=None):
    '''
        :param n_proc:多进程
        :return: 均值stoi, 均值pesq,均值ssnr
    '''

    q_score = Queue(buffersize)
    q_task = Queue(buffersize)
    q_summary = Queue(buffersize)

    n_workers = max(n_proc - 1, 1)
    summary_process = _SummaryRunner(q_score, q_summary, n_workers, samplerate)
    summary_process.start()

    for _ in range(n_workers):
        p_eval = _EvalCaculateRunner(q_task, q_score, samplerate)
        p_eval.start()

    for i in range(len(x)):
        z, _ = net.forward(Variable(x[i].unsqueeze(0)).type_as(next(net.parameters())))
        enhed = z[0].data.cpu().numpy()
        enhed = enhed / max(abs(enhed)+EPS)*0.999
        voice = y[i]
        voice = voice / max(abs(voice) + EPS) * 0.999
        mix = raw_mix[i] if raw_mix else x[i]
        mix = np.array(mix)
        mix = mix / max(1.1 * abs(mix))
        q_task.put([voice, enhed, mix])
    q_task.put(None)

    summary_process.join()
    mean_stoi, mean_pesq, mean_ssnr = q_summary.get()
    return mean_stoi, mean_pesq, mean_ssnr


class _EvalCaculateRunner(Process):
    def __init__(self, in_sample_queue, out_sample_queue, samplerate):
        super().__init__()
        self.in_sample_queue = in_sample_queue
        self.out_sample_queue = out_sample_queue
        self.samplerate = samplerate

    def run(self):
        while True:
            result = self.in_sample_queue.get()
            if result is None:
                self.out_sample_queue.put(None)
                self.in_sample_queue.put(None)
                break
            else:
                clean_raw, enhed_raw, mix = result
                stoi = assess.calc_stoi(clean_raw, enhed_raw, self.samplerate)
                pesq = assess.calc_pesq(clean_raw, enhed_raw, self.samplerate, False)
                ssnr = assess.calc_ssnr(clean_raw, enhed_raw, 1024)
                # ssnr = 0
                self.out_sample_queue.put([stoi, pesq, ssnr, mix, enhed_raw])


class _SummaryRunner(Process):
    def __init__(self, in_sample_queue, out_summary_queue, n_workers, samplerate):
        super().__init__()
        self.sample_queue = in_sample_queue
        self.summary_queue = out_summary_queue
        self.n_workers = n_workers
        self.samplerate = samplerate

    def run(self):
        mean_stoi, mean_pesq, mean_ssnr = 0., 0., 0.
        n_finished, count = 0, 0
        r = np.array([])
        while True:
            sample_score = self.sample_queue.get()
            if sample_score is None:
                n_finished += 1
                if n_finished >= self.n_workers:
                    break
                continue
            stoi, pesq, ssnr, mix, enhed = sample_score

            count += 1
            mean_stoi += stoi
            mean_pesq += pesq
            mean_ssnr += ssnr

            mix = util.norm_wav(mix)
            enhed = util.norm_wav(enhed)
            r = np.hstack((r, mix, enhed))
        print('writing out test audio. max %g, min %g, dtype %s' % (max(r), min(r), r.dtype))
        util.wav_write('tsep_mse.wav', r, self.samplerate, norm=True)
        mean_stoi, mean_pesq, mean_ssnr = mean_stoi / count, mean_pesq / count, mean_ssnr / count
        self.summary_queue.put([mean_stoi, mean_pesq, mean_ssnr])
