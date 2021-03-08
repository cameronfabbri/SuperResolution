"""

"""
import os
import time

import torch
import src.networks as networks
import torchvision.transforms as transforms
import copy
import gc

from src.videodata import VideoDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class Train:

    def __init__(self, args):

        self.num_ds = args.num_ds
        self.lambda_l1 = args.lambda_l1
        self.batch_size = args.batch_size
        self.num_blocks = args.num_blocks
        self.block_type = args.block_type
        self.model_dir = args.model_dir
        self.test_training = args.test

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if args.block_type == 'sisr':
            resblocks = networks.SISR_Resblocks(args.num_blocks)
        elif args.block_type == 'rrdb':
            resblocks = networks.RRDB_Resblocks(args.num_blocks)

        self.net_g = networks.Generator(resblocks).to(self.device)
        self.l1_loss = torch.nn.L1Loss()

        self.step = 0
        self.optimizer = torch.optim.Adam(
            self.net_g.parameters(), lr=args.lr_g)

        # If we're resuming a training session, this is the place to do it
        if args.resume_training:
            self.load()

        self.resize_func = transforms.Resize(args.patch_size)
        self.resize2_func = transforms.Resize((1080, 1440))

        self.train_dir = os.path.join(args.data_dir, 'train')
        self.test_dir = os.path.join(args.data_dir, 'test')

        self.train_dataset = VideoDataset(
            root_dir=self.train_dir,
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

        if self.test_training:
            self.static_test_batch = []
            self.get_test_frame(args.patch_size, args.num_ds)

    def load(self):
        checkpoint = torch.load(os.path.join(self.model_dir, 'model.pth'))
        self.net_g.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        self.step = checkpoint['step_num']
        print("Loaded saved model at step {} and set model/optim states".format(checkpoint['step_num']))

    def save(self):
        print("trying to save model...")

        print("step num is", self.step)
        model_state_dict = self.net_g.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        torch.save({
            'step_num': self.step,
            'model_state': model_state_dict,
            'optim_state': optim_state_dict
        }, os.path.join(self.model_dir, 'model.pth'))

    def step_g(self, batch_x, batch_y):

        self.optimizer.zero_grad()

        batch_g = self.net_g(batch_x)
        loss = self.lambda_l1 * self.l1_loss(batch_g, batch_y)

        loss.backward()

        self.optimizer.step()

        return loss, batch_g

    # initialize our dataloader here because it caches it's cache_size in ram, and we don't want that hanging around
    def get_test_frame(self, patch_size, num_ds):
        self.test_dataset = VideoDataset(
            root_dir=self.test_dir,
            train=False,
            cache_size=0.5,
            patch_size=patch_size,
            num_ds=num_ds)
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False)

        # Get 1 frame
        for batch_y in self.test_data_loader:
            #self.static_test_batch.append(batch_y.to(self.device))
            self.static_test_batch.append(batch_y)
            break


    # This is in a separate function so we can create (and destroy) our net on the cpu so
    # that we don't blast video ram with a full image convolution
    def test(self):
        i = 0
        print('Saving out test images')
        # Copy the net over to our cpu so we don't blast our vram
        cpunet_g = copy.deepcopy(self.net_g).to('cpu')
        print("Duplicated model on cpu...")
        for test_batch_y in self.static_test_batch:
            _, _, yh, yw = test_batch_y.shape
            x_size = (yh // self.num_ds, yw // self.num_ds)
            resize_func = transforms.Resize(x_size)
            test_batch_x = resize_func(test_batch_y)

            #test_batch_g = self.net_g(test_batch_x)
            test_batch_g = cpunet_g(test_batch_x)

            for bx, by, bg in zip(test_batch_x, test_batch_y, test_batch_g):
                #bx = (bx + 1.) / 2.
                #by = (by + 1.) / 2.
                #bg = (bg + 1.) / 2.
                bx = self.resize2_func(bx)
                canvas = torch.cat([bx, by, bg], axis=1)
                save_image(
                    canvas, os.path.join('test', str(self.step).zfill(3)+'_'+str(i)+'.png'))
                break
            i += 1
            break


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

                # test our model
                if self.test_training:
                    self.test()

                # Save our model
                self.save()
                

                

            # make sure our superlist is indeed empty
            assert self.train_dataset.data_decoder.get_num_chunks() == 0, "Still have some chunks left unloaded"
            # Rebuild our superlist and send er again
            self.train_dataset.data_decoder.build_chunk_superlist()
