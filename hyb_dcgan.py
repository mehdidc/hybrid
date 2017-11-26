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
from folder import ImageFolder

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
    def __init__(self, nz=100, nc=3, nb_classes=10, ngf=64, image_size=32):
        super().__init__()
        self.nz = nz
        self.nb_classes = nb_classes
        self.nc = nc
        self.ngf = ngf
        if image_size == 32:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=True),
            )
        elif image_size == 64:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            )

    def forward(self, input):
        return self.main(input)

class _netD(nn.Module):
    def __init__(self, nc=3, nb_classes=10, ndf=64, no=1, image_size=32):
        super().__init__()
        self.nc = nc
        self.nb_classes = nb_classes
        self.ndf = ndf
        self.no = no
        if image_size == 32:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, no, 4, 1, 0, bias=True),
            )
        elif image_size == 64:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, no, 4, 1, 0, bias=True),
            )

    def forward(self, input):
        output = self.main(input)
        return output.view(output.size(0), output.size(1))


def train(*, dataset='mnist', dataroot='.', workers=1, batch_size=64, image_size=32, 
          nz=100, ncols=3, nb_classes=10, ngf=64, ndf=64, iter=25, lr=0.0002, beta1=0.5, 
          classes=None, cuda=False, netG='', netD='', niter=100, outf='out'):
    try:
        os.makedirs(outf)
    except OSError:
        pass
    cudnn.benchmark = True
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if dataset == 'folder':
        dataset = ImageFolder(
            root=dataroot,
            valid_classes=classes,
            transform=transforms.Compose([
                transforms.Scale(image_size),
                transforms.CenterCrop(image_size),
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
    netG = _netG(nz=nz, nc=ncols, nb_classes=nb_classes, ngf=ngf, image_size=image_size) 
    netD = _netD(nc=ncols, nb_classes=nb_classes, ndf=ndf, no=nb_classes + 1, image_size=image_size)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(batch_size, 3, image_size, image_size)
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    fixed_noise = torch.randn(batch_size, nz, 1, 1)
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

    aux_criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(niter):
        for i, data in enumerate(dataloader):

            netD.zero_grad()
            real_cpu, real_classes = data
            if ncols == 1:
                real_cpu = real_cpu[:, 0:1]

            real_classes = real_classes.long().view(-1, 1)
            real_classes_var = Variable(real_classes[:, 0])
            batch_size = real_cpu.size(0)
            
            y_onehot = torch.zeros(batch_size, nb_classes)
            y_onehot.scatter_(1, real_classes, 1)

            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)
    
            output = netD(input)
            errD_real = (
                criterion(nn.Sigmoid()(output[:, 0]), label) + 
                aux_criterion((output[:, 1:]), real_classes_var)
            )

            _, pred = output[:, 1:].max(1)
            acc_real = torch.mean((pred.data.cpu() == real_classes[:, 0]).float())

            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            z = torch.randn(batch_size, nz, 1, 1)
            noise.data.resize_(z.size()).copy_(z)
            fake = netG(noise)
            fake = gen_activation(fake)
            
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = (
                criterion(nn.Sigmoid()(output[:, 0]), label) +
                aux_criterion((output[:, 1:]), real_classes_var)
            )

            _, pred = output[:, 1:].max(1)
            acc_fake = torch.mean((pred.data.cpu() == real_classes[:, 0]).float())

            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake)
            pr = nn.Softmax()(output[:, 1:])
            score = output[:, 1:].mean()
            entropy = -(pr * torch.log(pr + 1e-10)).sum(1).mean()
            errG = (
                criterion(nn.Sigmoid()(output[:, 0]), label) - 
                aux_criterion((output[:, 1:]), real_classes_var)
                #-10*entropy
            )
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            
            print('[{}/{}][{}/{}] acc_real : {:.4f} acc_fake : {:.4f} entropy_fake : {:.4f}'.format(
                  epoch, niter, i, len(dataloader), acc_real, acc_fake, entropy.data[0]))
            if i % 100 == 0:
                # the first 64 samples from the mini-batch are saved.
                im = real_cpu[0:64]
                if ncols == 3:
                    im = (im + 1) / 2.
                vutils.save_image(im, '%s/real_samples.png' % outf)
                fake = netG(fixed_noise)
                fake = gen_activation(fake)
                im = fake.data
                if ncols == 3:
                    im = (im + 1) / 2.
                fname = '%s/fake_samples_epoch_%03d.png' % (outf, epoch)
                vutils.save_image(im, fname)
        # do checkpointing
        torch.save(netG, '%s/netG.th' % (outf,))
        torch.save(netD, '%s/netD.th' % (outf,))


def gen(*, folder='', nb=100, out='out.png'):
    filename = '{}/netG.th'.format(folder)
    netG = torch.load(filename)
    filename = '{}/netD.th'.format(folder)
    netD = torch.load(filename)

    nz = netG.nz
    nb_classes = netG.nb_classes
    ncols = netG.nc
    z = torch.randn(nb, nz, 1, 1)
    z = Variable(z)
    x = netG(z)
    gen_activation = nn.Tanh() if ncols == 3 else nn.Sigmoid()
    x = gen_activation(x)

    y = netD(x)
    y = nn.Softmax()(y)
    print(y)

    im = x.data
    if ncols == 3:
        im = (im + 1) / 2.
    #im = im > 0.5
    im = im.numpy()
    im = grid_of_images_default(im)
    imsave(out, im)

if __name__ == '__main__':
    run([train, gen])
