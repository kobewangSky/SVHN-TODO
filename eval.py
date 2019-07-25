import argparse
import os

import torch
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader


def _eval(path_to_checkpoint: str, path_to_data_dir: str, path_to_results_dir: str):
    os.makedirs(path_to_results_dir, exist_ok=True)

    dataset = Dataset(path_to_data_dir, mode= Dataset.Mode.TEST)
    dataloader = DataLoader(dataset, batch_size= 64)

    mode = Model().cuda()
    mode.load(path_to_checkpoint)

    num_hits = 0
    AllIndex = 0
    print('Start evaluating')

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(dataloader):
            images = images.view(-1, 3, 64, 64).float().cuda()
            labels = labels.cuda()

            logits = mode.eval().forward(images)
            temp = 0
            for i in range(len(logits)):
                _, predictions = logits[i].max(dim=1)
                temp += (predictions == labels[:, i]).sum().item()
            temp = temp / (len(labels) * 6)
            num_hits += temp
            AllIndex = AllIndex +1

        accuracy = num_hits / AllIndex
        print(f'Accuracy = {accuracy:.4f}')

    with open(os.path.join(path_to_results_dir, 'accuracy.txt'), 'w') as fp:
        fp.write(f'{accuracy:.4f}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_results_dir = args.results_dir

        _eval(path_to_checkpoint, path_to_data_dir, path_to_results_dir)

    main()
