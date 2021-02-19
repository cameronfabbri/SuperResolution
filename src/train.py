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

        self.dataset = VideoDataset(root_dir='data/train', train=True, cache_size=20)
        self.data_loader = DataLoader(
                self.dataset,
                batch_size=7,
                shuffle=True,
                num_workers=0, 
                #prefetch_factor=4,
                #persistent_workers=True,
                pin_memory=True
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

        num_epochs = self.dataset.get_epochs_per_dataset()
        print("{} epochs to loop over entire dataset once".format(num_epochs))

        for loop in range(100): # how many times we're going to loop over our entire dataset

            for epoch in range(num_epochs):

                for batch_data in self.data_loader:
                    batch_x, batch_y = batch_data

                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # shift our data's range from [0, 1] to [-1, 1]
                    batch_x = (batch_x * 2.) - 1.
                    batch_y = (batch_y * 2.) - 1.

                    s = time.time()
                    loss, batch_g = self.step_g(batch_x, batch_y)
                    loss = round(loss.cpu().item(), 3)
                    self.step += 1

                    #print('| epoch:', epoch, '/', num_epochs, '| step:', self.step, '| loss:', loss, '| time:', round(time.time()-s, 2))
                    print("| itr:{} | epoch:{}/{} | step:{} | loss:{} | time:{} |".
                          format(loop, epoch, num_epochs, self.step, loss, round(time.time()-s, 2)))
                    # for debugging purposes, cap our epochs to whatever number of steps
                    #if not self.step % 50:
                    #    break

                # Swap memory cache and train on the new stuff
                self.dataset.swap()

                batch_x = (batch_x + 1.) / 2.
                batch_y = (batch_y + 1.) / 2.
                batch_g = (batch_g + 1.) / 2.

                batch_x = self.resize_func(batch_x)

                canvas = torch.cat([batch_x[:1], batch_y[:1], batch_g[:1]], axis=3)

                save_image(canvas[0], 'test/'+str(self.step).zfill(3) + '-' + str(loss).zfill(3) + '.png')
            
            # make sure our superlist is indeed empty
            assert self.dataset.data_decoder.get_num_chunks() != 0, "Still have some chunks left unloaded"
            # Rebuild our superlist and send er again
            self.dataset.data_decoder.build_chunk_superlist()


if __name__ == '__main__':
    t = Train()
    t.train()
