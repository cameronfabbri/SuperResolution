"""

"""
import time

import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.utils.data import DataLoader

import data as data
from videodata import VideoDataset
import networks as networks


class Train:

    def __init__(self):

        #dataset = data.AnimeHDDataset(root_dir='data/train', train=True)
        #self.data_loader = DataLoader(
        #        dataset, batch_size=7, shuffle=True, num_workers=4, persistent_workers=True
        #    )

        dataset = VideoDataset(root_dir='data/train', file_name='vid1.mkv', train=True)
        self.data_loader = DataLoader(
                dataset, batch_size=7, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=False
            )

        self.device = torch.device('cuda:0')

        self.net_g = networks.Generator().to(self.device)
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=1e-3)

        self.step = 0

        self.resize_func = transforms.Resize(512)

    def step_g(self, batch_x, batch_y):

        self.optimizer.zero_grad()

        batch_g = self.net_g(batch_x)
        loss = 100 * self.l1_loss(batch_g, batch_y)

        loss.backward()

        self.optimizer.step()

        return loss, batch_g

    def train(self):

        for epoch in range(1000):

            for batch_data in self.data_loader:
                batch_x, batch_y = batch_data

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_x = (batch_x * 2.) - 1.
                batch_y = (batch_y * 2.) - 1.

                s = time.time()
                loss, batch_g = self.step_g(batch_x, batch_y)
                loss = round(loss.cpu().item(), 3)
                self.step += 1

                print('| epoch:', epoch, '| step:', self.step, '| loss:', loss, '| time:', round(time.time()-s, 2))
                # for debugging purposes, cap our epochs to whatever number of steps
                if not self.step % 300:
                    break

            batch_x = (batch_x + 1.) / 2.
            batch_y = (batch_y + 1.) / 2.
            batch_g = (batch_g + 1.) / 2.

            batch_x = self.resize_func(batch_x)

            canvas = torch.cat([batch_x[:1], batch_y[:1], batch_g[:1]], axis=3)

            save_image(canvas[0], 'test/'+str(self.step).zfill(3) + '-' + str(loss).zfill(3) + '.png')


if __name__ == '__main__':
    t = Train()
    t.train()
