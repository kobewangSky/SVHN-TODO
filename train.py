import argparse
import os
import time
from collections import deque
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
from torch import optim
from torch.optim.lr_scheduler import StepLR
from DataAugment import DataAugment
import torch

from imgaug import augmenters as iaa


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay= 0.0005 )
    scheduler = StepLR(optimizer, step_size=150000, gamma=0.1)

    num_steps_to_display = 20
    num_steps_to_snapshot = 30000
    num_steps_to_finish = 300000

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=1000)
    should_stop = False

    print('Start training')

    aug = iaa.SomeOf(2, [
        iaa.Add((-20, 20)),
        iaa.ContrastNormalization((0, 1)),
        iaa.GaussianBlur(sigma=(0, 1)),  # blur images with a sigma of 0 to 3.0
        iaa.AdditiveGaussianNoise(scale=(0, 6))
    ])

    while not should_stop:
       for batch_idx, (images, labels) in enumerate(dataloader):

            tempimage = images.numpy()
            images_aug = aug.augment_images(tempimage)
            images = torch.from_numpy(images_aug)
            images = images.view(-1, 3, 64, 64).float().cuda()
            labels = labels.cuda()

            logits = model.forward(images)
            loss = model.loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec),  Learning Rate = {lr}')
            if step % num_steps_to_snapshot == 0:
                path_to_checkpoints = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoints_dir}')

            if step == num_steps_to_finish:
                should_stop = True
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
