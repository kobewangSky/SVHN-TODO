import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch import optim


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        hid1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        hid2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        hid3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        hid4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        hid5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        hid6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        hid7 = nn.Sequential(
            nn.Linear(1024 , 1024),
            nn.ReLU()

        )
        hid8 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.features = nn.Sequential(
            hid1,
            hid2,
            hid3,
            hid4,
            hid5,
            hid6,
        )

        self.classfi = nn.Sequential(
            hid7,
            hid8
        )

        self.digit_len = nn.Sequential(nn.Linear(1024, 7))
        self.digit1 = nn.Sequential(nn.Linear(1024, 10))
        self.digit2 = nn.Sequential(nn.Linear(1024, 10))
        self.digit3 = nn.Sequential(nn.Linear(1024, 10))
        self.digit4 = nn.Sequential(nn.Linear(1024, 10))
        self.digit5 = nn.Sequential(nn.Linear(1024, 10))





    def forward(self, images: Tensor) -> Tensor:

        x = self.features(images)
        x = x.view(-1, 1024)
        x = self.classfi(x)

        digit_len = self.digit_len(x)
        digit1 = self.digit1(x)
        digit2 = self.digit2(x)
        digit3 = self.digit3(x)
        digit4 = self.digit4(x)
        digit5 = self.digit5(x)

        return digit_len, digit1, digit2, digit3, digit4, digit5



    def loss(self, logits: Tensor, labels: Tensor) -> Tensor :

        Loss = F.cross_entropy(logits[0], labels[:, 0])
        for i in range(1, len(logits)):
            temp = F.cross_entropy(logits[i], labels[:, i])
            Loss = Loss + temp
        return Loss

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
