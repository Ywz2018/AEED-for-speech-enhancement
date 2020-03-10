# -*- coding: utf-8 -*-
'''
继承自aet_refinenet，实现aet相位幅度分离
'''
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy.random as random
from util import aet_util
from old_src import prep_data_v5 as pdata
import torchvision.models
import torch.nn as nn
import config

import warnings
# 关闭警告
warnings.filterwarnings("ignore")


def myloss(out, targ, type='mse'):
    return torch.mean(torch.pow(out - targ, 2.))


# 模型
class ScoreEstimator(nn.Module):
    # Initialize
    def __init__(self):
        super(ScoreEstimator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(502*8, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
            # nn.Sigmoid()
        )

    def forward(self, mix, voice):
        diff = torch.abs(mix) - torch.abs(voice)
        torchvision.models.alexnet()

        inp = torch.stack((mix, voice, diff))
        x = torch.transpose(inp, 0, 1)  # bt x 1 x T

        x = self.features(x)
        x = x.view(x.size(0), 502*8)
        x = self.classifier(x)
        return x[:, 0], x[:, 1] * 5 - 0.5

    def calc_loss(self, mix, voice):
        stoi, pesq = self.forward(mix, voice)
        return torch.mean(stoi), torch.mean(pesq)


def calc_loss(loss_nn, mix, voice):
    stoi, pesq = loss_nn.forward(mix, voice)
    return torch.mean(stoi), torch.mean(pesq)


def evaluate(data, net):
    loss_stoi, loss_pesq = 0, 0
    count = 0
    for mix, voice, stoi, pesq in data:
        mix = Variable(mix.unsqueeze(0)).type_as(next(net.parameters()))
        voice = Variable(voice.unsqueeze(0)).type_as(next(net.parameters()))
        e_stoi, e_pesq = net.forward(mix, voice)
        loss_stoi += myloss(e_stoi, stoi)
        loss_pesq += myloss(e_pesq, pesq)
        count += 1
    return loss_stoi / max(1, count), loss_pesq / max(1, count)


def train_wav(model, dataloader, start_time):
    params = model.parameters()
    opt2 = torch.optim.Adam(params, lr=config.TRAIN_WAV_LR)

    dataset = dataloader.dataset
    xv = dataset.getvals()
    xv_tr = dataset.getvals(data='subtrain')

    last_eval_time = time.time()
    last_epoch, loss_sum, count = 0, 0., 0
    epoch = 0
    while epoch < config.UNION_TRAIN_ITER:
        for mix, voice, stoi, pesq in dataloader:
            mix = Variable(mix).type_as(next(model.parameters()))
            voice = Variable(voice).type_as(next(model.parameters()))
            stoi = Variable(stoi).type_as(next(model.parameters()))
            pesq = Variable(pesq).type_as(next(model.parameters()))

            # Get loss
            model.train()
            e_stoi, e_pesq = model(mix, voice)
            loss = myloss(e_stoi, stoi, type=config.EST_SCORE_LOSS)
            loss += myloss(e_pesq, pesq, type=config.EST_SCORE_LOSS) / 5
            loss_sum += float(loss)
            count += 1

            # Update
            opt2.zero_grad()
            loss.backward()
            opt2.step()

            # Report
            model.eval()

        _, _, _, time_str = aet_util.sec2hms(time.time() - start_time)
        print("Epoch: %.4g, loss: %.4g, elapse: %s" % (epoch, loss_sum / max(1, count), time_str))
        loss_sum, count = 0., 0

        # Test on validation data
        if time.time() - last_eval_time > config.TICK or epoch + 1 >= config.UNION_TRAIN_ITER:
            print('testing...')
            loss_stoi, loss_pesq = evaluate(xv, model)
            s_loss_stoi, s_loss_pesq = evaluate(xv_tr, model)
            metrics_str = "loss: stoi=%.4f, pesq=%.4f " % (loss_stoi, loss_pesq)
            metrics_str_tr = "test on sub-training dataset, loss: stoi: %.4f, pesq: %.4f" % \
                             (s_loss_stoi, s_loss_pesq)
            last_eval_time = time.time()
            _, _, _, time_str = aet_util.sec2hms(last_eval_time - start_time)
            torch.save(model.state_dict(), 'model/score-predictor.pt')
            print("elapse: %s, %s" % (time_str, metrics_str))
            print(metrics_str_tr)

        epoch += 1

    torch.save(model.state_dict(), 'model/score-predictor.pt')


def main():
    model = ScoreEstimator()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    # Select data
    random.seed(25)
    dataset = pdata.Mix_Dataset()
    dataloader = DataLoader(dataset, batch_size=config.BATCHSIZE, num_workers=config.NUM_WORKERS,
                            pin_memory=True, shuffle=True)

    # 加载已保存模型参数
    model.load_state_dict(torch.load('model/score-predictor.pt'))
    model.eval()

    start_time = time.time()

    print()
    print('union train')
    train_wav(model, dataloader, start_time)


if __name__ == "__main__":
    main()
