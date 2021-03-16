"""

"""
import gc
import os
import copy
import time

import torch
import src.networks as networks
import torchvision.transforms as transforms

from src.videodata import VideoDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class Train:

    def build_gan_loss(self):

        criterion = torch.nn.BCEWithLogitsLoss()

        def loss_g(d_fake):
            y_real = torch.ones(d_fake.shape)
            loss = criterion(d_fake, y_real)
            return loss

        def loss_d(d_real, d_fake):
            y_real = torch.ones(d_fake.shape)
            y_fake = torch.zeros(d_fake.shape)

            loss_real = criterion(d_fake, y_real)
            loss_fake = criterion(d_fake, y_fake)

            return loss_real + loss_fake

        return loss_g, loss_d

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

        # define the generator and descriminiator
        self.net_g = networks.Generator(resblocks).to(self.device)
        self.net_d = networks.VGGStyleDiscriminator128().to(self.device)
        self.l1_loss = torch.nn.L1Loss()

        self.step = 0
        self.optimizer_g = torch.optim.Adam(
            self.net_g.parameters(), lr=args.lr_g)

        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(), lr=1e-3)

        # If we're resuming a training session, this is the place to do it
        if args.resume_training:
            self.load()

        self.loss_g, self.loss_d = build_gan_loss()

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
        self.optimizer_g.load_state_dict(checkpoint['optim_state'])
        self.step = checkpoint['step_num']
        print("Loaded saved model at step {} and set model/optim states".format(checkpoint['step_num']))

    def save(self):
        print("trying to save model...")

        print("step num is", self.step)
        model_state_dict = self.net_g.state_dict()
        optim_state_dict = self.optimizer_g.state_dict()
        torch.save({
            'step_num': self.step,
            'model_state': model_state_dict,
            'optim_state': optim_state_dict
        }, os.path.join(self.model_dir, 'model.pth'))

    # initialize our dataloader here because it caches it's cache_size in ram,
    # and we don't want that hanging around
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

    def checkpoint(self, input, gt, output):
        dims = gt.size()[3]

        input = transforms.Resize(dims)(input)
        # Bring this back to 0 to 1 from -1 to 1
        input = (input + 1.) / 2.
        gt = (gt + 1.) / 2.
        output = (output + 1.) / 2.
        canvas = torch.cat([input[:1], gt[:1], output[:1]], axis=3)
        save_image(canvas, os.path.join('test', str(self.step).zfill(3)+'.png'))

    def step_d(self, batch_truth, batch_lie):

        self.optimizer_d.zero_grad()

        # Make two tensors, one is a block of true, one a block of false, and its associated expected output
        all_data = torch.cat((batch_truth, batch_lie), 0)
        all_data_labels = torch.cat(
            # make an array of 1s and 0s, and use the batch size of the samples passed
            (torch.ones((batch_truth.size()[0], 1)).to(self.device),
            torch.zeros((batch_lie.size()[0], 1)).to(self.device)),
            0
        )

        # run our real and fake data through the descriminator
        descriminiator_output = self.net_d(all_data)
        # calculate the loss of the real images, just using dumb l1 for now to get a psnr-oriented model
        loss = self.lambda_l1 * self.l1_loss(descriminiator_output, all_data_labels)
        loss.backward()

        self.optimizer_d.step()

        return loss

    def step_g(self, batch_x, batch_y, discriminator=True):

        self.optimizer_g.zero_grad()

        batch_g = self.net_g(batch_x)

        l1_loss = self.lambda_l1 * self.l1_loss(batch_g, batch_y)

        total_loss = l1_loss

        # calculate loss
        if discriminator:
            # run our generated samples through our discriminator (not racist)
            real_tags = torch.ones((batch_y.size()[0], 1)).to(self.device)
            d_fake = self.net_d(batch_g)
            g_loss = self.lambda_gan * self.g_loss(d_fake)
            total_loss += g_loss

        total_loss.backward()

        self.optimizer_g.step()

        return loss, batch_g.detach() # detaches from the autograd in step_g so the tensor can be worked on

    def train(self):

        num_epochs = self.train_dataset.get_epochs_per_dataset()
        print("{} epochs to loop over entire dataset once".format(num_epochs))

        print("here we go aGAN ;)")
        # How many times we're going to loop over our entire dataset
        for loop in range(100):

            for epoch in range(num_epochs):

                for batch_data in self.train_data_loader:
                    batch_x, batch_y, batch_r = batch_data


                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_r = batch_r.to(self.device)

                    # shift our data's range from [0, 1] to [-1, 1] (should really move this to dataloader)
                    batch_x = (batch_x * 2.) - 1.
                    batch_y = (batch_y * 2.) - 1.

                    # start our batch timer
                    s = time.time()
                    # Train our generator with just l1 for a while
                    #if self.step < 1000:
                    #    loss_g, batch_g = self.step_g(batch_x, batch_y, discriminator=False)
                    #    #loss_d = self.step_d(batch_truth=batch_y, batch_lie=batch_g)
                    #    loss_d = loss_g
                    #else:
                    #    if self.step == 1000:
                    #        # reinitilize our optimizer so it doesnt' blast
                    #        print("RESTARTING OPTIMIZER")
                    #        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=1e-4)
                    #    loss_g, batch_g = self.step_g(batch_x, batch_y, discriminator=True)
                    #    loss_d = self.step_d(batch_truth=batch_y, batch_lie=batch_g)

                    loss_g, batch_g = self.step_g(batch_x, batch_y, discriminator=True)
                    loss_d = self.step_d(batch_truth=batch_y, batch_lie=batch_g)

                    self.step += 1

                    # round our losses
                    loss_d = round(loss_d.cpu().item(), 3)
                    loss_g = round(loss_g.cpu().item(), 3)
                    # print our update
                    print("| itr:{} | epoch:{}/{} | step:{} | d_loss:{} | g_loss:{} | time:{} |".
                          format(loop, epoch, num_epochs, self.step, loss_d, loss_g, round(time.time()-s, 2)))
                    # for debugging purposes, do whatever X amount of steps
                    if not self.step % 100:
                        print("checkpoint!")
                        self.checkpoint(batch_x, batch_y, batch_g)
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
