from __future__ import print_function
import numpy as np
from skimage.io import imsave
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage.transform import resize
from torch.autograd import Variable
from machinedesign.viz import grid_of_images_default
from torchsample.datasets import TensorDataset

from clize import run

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, nz=100, nc=3, nb_classes=10, ngf=64):
        super().__init__()
        self.nz = nz
        self.nb_classes = nb_classes
        self.nc = nc
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nb_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf * 2,      nc, 4, 2, 1, bias=False),
            #nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)

class _netD(nn.Module):
    def __init__(self, nc=3, nb_classes=10, ndf=64):
        super().__init__()
        self.nc = nc
        self.nb_classes = nb_classes
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + nb_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


def train(*, dataset='mnist', dataroot='.', workers=1, batch_size=64, image_size=32, 
          nz=100, ncols=3, nb_classes=10, ngf=64, ndf=64, iter=25, lr=0.0002, beta1=0.5, 
          cuda=False, netG='', netD='', niter=100, outf='out'):
    try:
        os.makedirs(outf)
    except OSError:
        pass
    cudnn.benchmark = True
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Scale(image_size),
                #transforms.CenterCrop(image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * ncols if ncols == 3 else (0,), (0.5,) * ncols if ncols == 3 else (1,)),
        ]))
    elif dataset == 'mnist':
        dataset = dset.MNIST(
            root=dataroot, 
            download=True,
            transform=transforms.Compose([
               transforms.Scale(image_size),
               transforms.ToTensor(),
               #transforms.Normalize((0.5,), (0.5,)),
            ])
        )
    elif dataset == 'cifar10':
        dataset = dset.CIFAR10(
            root=dataroot, 
            download=True,
            transform=transforms.Compose([
                transforms.Scale(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, 
        num_workers=int(workers))
    netG = _netG(nz=nz, nc=ncols, nb_classes=nb_classes, ngf=ngf) 
    netD = _netD(nc=ncols, nb_classes=nb_classes, ndf=ndf)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(batch_size, 3, image_size, image_size)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    nb_rows = 10
    fixed_z = torch.randn(nb_rows, nb_classes, nz, 1, 1)
    fixed_z = fixed_z.view(nb_rows * nb_classes, nz, 1, 1)
    fixed_onehot = torch.zeros(nb_rows, nb_classes, nb_classes, 1, 1)
    fixed_onehot = fixed_onehot.view(nb_rows * nb_classes, nb_classes, 1, 1)
    for i in range(fixed_onehot.size(0)):
        cl = i % nb_classes
        fixed_onehot[i, cl] = 1
    fixed_noise = torch.cat((fixed_z, fixed_onehot), 1)
    if cuda:
        fixed_noise = fixed_noise.cuda()
    label = torch.FloatTensor(batch_size)
    real_label = 1
    fake_label = 0

    if cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))
    gen_activation = nn.Tanh() if ncols == 3 else nn.Sigmoid()

    for epoch in range(niter):
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, real_classes = data
            real_cpu = real_cpu[:, 0:ncols, :, :]

            real_classes = real_classes.long().view(-1, 1)
            batch_size = real_cpu.size(0)
            
            y_onehot = torch.zeros(batch_size, nb_classes)
            y_onehot.scatter_(1, real_classes, 1)
            y_onehot_ = y_onehot
            y_onehot = y_onehot.view(y_onehot.size(0), y_onehot.size(1), 1, 1)
            y_onehot = y_onehot.repeat(1, 1, real_cpu.size(2), real_cpu.size(3))
            real_cpu_with_class = torch.cat((real_cpu, y_onehot), 1)

            input.data.resize_(real_cpu_with_class.size()).copy_(real_cpu_with_class)
            label.data.resize_(batch_size).fill_(real_label)
    
            output = netD(input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            z = torch.randn(batch_size, nz, 1, 1)
            z = torch.cat((z, y_onehot_), 1)
            noise.data.resize_(z.size()).copy_(z)
            fake = netG(noise)
            fake = gen_activation(fake)
            v = Variable(y_onehot)
            if cuda:
                v = v.cuda()
            fake_with_class = torch.cat( (fake, v), 1)

            label.data.fill_(fake_label)
            output = netD(fake_with_class.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake_with_class)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                # the first 64 samples from the mini-batch are saved.
                im = real_cpu[0:64]
                if ncols == 3:
                    im = (im + 1) / 2.
                vutils.save_image(im, '%s/real_samples.png' % outf, nrow=8)
                fake = netG(fixed_noise)
                fake = gen_activation(fake)
                im = fake.data
                if ncols == 3:
                    im = (im + 1) / 2.
                fname = '%s/fake_samples_epoch_%03d.png' % (outf, epoch)
                vutils.save_image(im, fname, nrow=nb_classes)
        # do checkpointing
        torch.save(netG, '%s/netG.th' % (outf,))
        torch.save(netD, '%s/netD.th' % (outf,))


def gen(*, gen='', classes='0', nb=9, out='out.png'):
    netG = torch.load(gen)
    nz = netG.nz
    nb_classes = netG.nb_classes
    ncols = netG.nc
    classes = classes.split(',')
    classes = map(int, classes)
    classes = list(classes)
    
    z = torch.randn(nb, nz, 1, 1)
    onehot = torch.zeros(nb, nb_classes, 1, 1)
    for cl in classes:
        onehot[:, cl] = 1.0 / len(classes)
    z = torch.cat((z, onehot), 1)
    z = Variable(z)
    x = netG(z)
    gen_activation = nn.Tanh() if ncols == 3 else nn.Sigmoid()
    x = gen_activation(x)
    im = x.data
    if ncols == 3:
        im = (im + 1) / 2.
    #im = im > 0.5
    im = im.numpy()
    im = grid_of_images_default(im)
    imsave(out, im)

if __name__ == '__main__':
    run([train, gen])
