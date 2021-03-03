"""

"""
import os
import time

import torch
import src.networks as networks
import torchvision.transforms as transforms

from src.videodata import VideoDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class Train:

    def __init__(self, args):

        self.batch_size = args.batch_size
        self.lambda_l1 = args.lambda_l1
        self.num_ds = args.num_ds

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        train_dir = os.path.join(args.data_dir, 'train')
        test_dir = os.path.join(args.data_dir, 'test')

        self.train_dataset = VideoDataset(
            root_dir=train_dir,
            train=True,
            cache_size=args.cache_size,
            patch_size=args.patch_size,
            num_ds=args.num_ds)
        self.train_data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True)

        self.test_dataset = VideoDataset(
            root_dir=test_dir,
            train=False,
            cache_size=args.cache_size,
            patch_size=args.patch_size,
            num_ds=args.num_ds)
        self.test_data_loader = DataLoader(
                self.test_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False)

        self.static_test_batch = []
        for batch_y in self.test_data_loader:
            self.static_test_batch.append(batch_y.to(self.device))
            break

        self.net_g = networks.Generator().to(self.device)
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(
            self.net_g.parameters(), lr=args.lr_g)

        self.step = 0
        self.resize_func = transforms.Resize(args.patch_size)
        self.resize2_func = transforms.Resize((1080, 1440))

    def step_g(self, batch_x, batch_y):

        self.optimizer.zero_grad()

        batch_g = self.net_g(batch_x)
        loss = self.lambda_l1 * self.l1_loss(batch_g, batch_y)

        loss.backward()

        self.optimizer.step()

        return loss, batch_g

    def train(self):

        num_epochs = self.train_dataset.get_epochs_per_dataset()
        print("{} epochs to loop over entire dataset once".format(num_epochs))

        # How many times we're going to loop over our entire dataset
        for loop in range(100):

            for epoch in range(num_epochs):

                for batch_data in self.train_data_loader:
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
                self.train_dataset.swap()

                i = 0
                for test_batch_y in self.static_test_batch:
                    _, _, yh, yw = test_batch_y.shape
                    x_size = (yh // self.num_ds, yw // self.num_ds)
                    resize_func = transforms.Resize(x_size)
                    test_batch_x = resize_func(test_batch_y)

                    test_batch_g = self.net_g(test_batch_x)

                    for bx, by, bg in zip(test_batch_x, test_batch_y, test_batch_g):
                        bx = (bx + 1.) / 2.
                        by = (by + 1.) / 2.
                        bg = (bg + 1.) / 2.
                        bx = self.resize2_func(bx)
                        canvas = torch.cat([bx, by, bg], axis=1)
                        save_image(
                            canvas, os.path.join('test', str(self.step).zfill(3)+'_'+str(i)+'.png'))
                        break
                    i += 1

            # make sure our superlist is indeed empty
            assert self.dataset.data_decoder.get_num_chunks() == 0, "Still have some chunks left unloaded"
            # Rebuild our superlist and send er again
            self.dataset.data_decoder.build_chunk_superlist()
