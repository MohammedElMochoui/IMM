"""
Simple MNIST sanity check for the VAE

"""


import torch

from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Linear, Sequential, ReLU, Sigmoid, Softplus, Upsample
from torch.autograd import Variable
import torch.distributions as dist

import numpy as np
import ptutil, random, tqdm, math

from argparse import ArgumentParser

from tensorboardX import SummaryWriter

import torchvision
from torchvision import transforms

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

EPS = 0.00001

torch.cuda.current_device()

print(f"Torch cuda is: {torch.cuda.is_available()}")

def go(options):

    # os.system("some_command with args")

    tbw = SummaryWriter(log_dir=f"{options.tb_dir}")

    # Set random or det. seed
    if options.seed < 0:
        seed = random.randint(0, 1000000)
    else:
        seed = options.seed

    np.random.seed(seed)
    print('random seed: ', seed)

    # load data

    data_dir_train = "Dataset/train1"
    data_dir_test = "Dataset/test1"

    normalize = transforms.Compose([transforms.ToTensor()])

    train = torchvision.datasets.ImageFolder(root=data_dir_train, transform=normalize)
    trainloader = torch.utils.data.DataLoader(train, batch_size=options.batch_size, shuffle=True, num_workers=2)

    test = torchvision.datasets.ImageFolder(root=data_dir_test, transform=normalize)
    testloader = torch.utils.data.DataLoader(test, batch_size=options.batch_size, shuffle=False, num_workers=2)

    ## Load the complete test set into a listr of batches
    test_batches = [inputs for inputs, _ in testloader]
    test_images = torch.cat(test_batches, dim=0).numpy()

    test_batch = test_batches[0]
    assert test_batch.size(0) > 10
    if torch.cuda.is_available():
        test_batch = test_batch.cuda()

    ## Build model


    outc = 3 if options.loss == 'xent' else 2
    act = Softplus() if options.loss == 'beta' else Sigmoid()

    # - channel sizes
    edge_a, edge_b, edge_c = 8, 32, 128
    h, w = 256, 256

    edge_encoder = Sequential(
        Conv2d(3, edge_a, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(edge_a, edge_b, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(edge_b, edge_c, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        ptutil.Flatten(),
        Linear((h // 8) * (w // 8) * edge_c, 2 * options.latent_size)
    )

    upmode = 'bilinear'
    edge_decoder = Sequential(
        Linear(options.latent_size, edge_c * (h // 8) * (w // 8)), ReLU(),
        ptutil.Reshape((edge_c, 32, 32)),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(edge_c, edge_b, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(edge_b, edge_a, (5, 5), padding=2), ReLU(), # note the padding
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(edge_a, outc, (5, 5), padding=2), act
    )

    # - channel sizes

    handbag_encoder = Sequential(
        Conv2d(3, edge_a, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(edge_a, edge_b, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(edge_b, edge_c, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        ptutil.Flatten(),
        Linear((h // 8) * (w // 8) * edge_c, 2 * options.latent_size)
    )

    upmode = 'bilinear'
    handbag_decoder = Sequential(
        Linear(options.latent_size, edge_c * (h // 8) * (w // 8)), ReLU(),
        ptutil.Reshape((edge_c, 32, 32)),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(edge_c, edge_b, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(edge_b, edge_a, (5, 5), padding=2), ReLU(), # note the padding
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(edge_a, outc, (5, 5), padding=2), act
    )

    if torch.cuda.is_available():
        edge_encoder.cuda()
        edge_decoder.cuda()
        handbag_encoder.cuda()
        handbag_decoder.cuda()

    params = list(edge_encoder.parameters()) + list(edge_decoder.parameters() + handbag_encoder.parameters() + list(handbag_decoder.parameters())

    ### Fit model
    instances_seen = 0

    optimizer = Adam(params, lr=options.lr)

    for epoch in range(options.epochs):
        for i, data in tqdm.tqdm(enumerate(trainloader, 0)):
            # if i > 1000:
            #     break

            # get the inputs
            inputs, _ = data

            edge = inputs[:, :, :, :256]
            handbag = inputs[:, :, :, 256:]

            b, c, w, h = edge.size()

            if torch.cuda.is_available():
                edge, handbag = edge.cuda(), handbag.cuda()

            # wrap them in Variables
            edge, handbag = Variable(edge), Variable(labels)

            optimizer.zero_grad()

            # Forward pass

            zcomb = edge_encoder(edge)
            zmean, zlsig = zcomb[:, :options.latent_size], zcomb[:, options.latent_size:]

            zcomb_handbag = handbag_encoder(handbag)
            zmean_handbag, zlsig_handbag = zcomb_handbag[:, :options.latent_size], zcomb_handbag[:, options.latent_size:]

            kl_loss = ptutil.kl_loss(zmean, zlsig)
            kl_loss_handbag = ptutil.kl_loss(zmean_handbag, zlsig_handbag)

            zsample = ptutil.sample(zmean, zlsig)
            zsample_handbag = ptutil.sample(zmean_handbag, zlsig_handbag)

            out = edge_decoder(zsample)
            out = handbag_decoder(zsample_handbag)

            if options.loss == 'xent':
                rec_loss = binary_cross_entropy(out, edge, reduce=False).view(b, -1).sum(dim=1) + \
                binary_cross_entropy(out_handbag, handbag, reduce=False).view(b, -1).sum(dim=1) + \
                binary_cross_entropy(out, handbag, reduce=False).view(b, -1).sum(dim=1) + \
                binary_cross_entropy(out_handbag, edge, reduce=False).view(b, -1).sum(dim=1) 
            else:
                raise Exception('loss {} not recognized'.format(options.loss))

            # Backward pass
            loss = (rec_loss + options.beta * kl_loss + options.beta * kl_loss_handbag).mean()
            loss.backward()

            optimizer.step()

            instances_seen += edge.size(0)

            tbw.add_scalar('score/kl', float(kl_loss.mean()), instances_seen)
            tbw.add_scalar('score/rec', float(rec_loss.mean()), instances_seen)
            tbw.add_scalar('score/loss', float(loss), instances_seen)
            tbw.add_scalar('score/bits', float(loss / 784), instances_seen)
            
            torch.save(edge_encoder.state_dict(), './edge_encoder_vae.pt')
            torch.save(edge_decoder.state_dict(), './edge_decoder_vae.pt')
            torch.save(handbag_encoder.state_dict(), './handbag_encoder_vae.pt')
            torch.save(handbag_decoder.state_dict(), './handbag_decoder_vae.pt')


        ## Plot some reconstructions
        if epoch % options.out_every == 0:

            print("Model saved!")
    
            print('({}) Plotting reconstructions.'.format(epoch))

            plt.figure(figsize=(10, 4))

            test_edge, test_handbag = test_batch[:, :, :, :256], test_batch[:, :, :, 256:]

            zc = edge_encoder(test_edge)
            zc_handbag = handbag_encoder(test_handbag)

            zmean, zlsig = zc[:, :options.latent_size], zc[:, options.latent_size:]
            zmean_handbag, zlsig_handbag = zc_handbag[:, :options.latent_size], zc_handbag[:, options.latent_size:]

            zsample = ptutil.sample(zmean, zlsig)
            zsample_handbag = ptutil.sample(zmean_handbag, zlsig_handbag)

            out = edge_decoder(zsample)
            out_handbag = handbag_decoder(zsample_handbag)

            for i in range(10):
                ax = plt.subplot(4, 10, i + 1)
                ax.imshow(np.moveaxis(test_edge[i].cpu().numpy(), 0, -1)) #, cmap='gray'
                ptutil.clean(ax)

                ax = plt.subplot(4, 10, i + 11)
                ax.imshow(np.moveaxis(out[i].cpu().detach().numpy(), 0, -1))
                ptutil.clean(ax)
                            
                ax = plt.subplot(4, 10, i + 21)
                ax.imshow(np.moveaxis(test_handbag[i].cpu().numpy(), 0, -1)) #, cmap='gray'
                ptutil.clean(ax)

                ax = plt.subplot(4, 10, i + 31)
                ax.imshow(np.moveaxis(out_handbag[i].cpu().detach().numpy(), 0, -1))
                ptutil.clean(ax)

            # plt.tight_layout()
            plt.savefig('plots/{}/rec.{:03}.pdf'.format(options.loss, epoch), dpi=300)

            # Clear the current axes.
            plt.cla() 
            # Clear the current figure.
            plt.clf() 
            # Closes all the figure windows.
            plt.close('all')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=50, type=int)

    parser.add_argument("-o", "--out-every",
                        dest="out_every",
                        help="Output every x epochs.",
                        default=1, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="latent_size",
                        help="Size of the latent representation",
                        default=256, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Batch size",
                        default=128, type=int)

    parser.add_argument("--loss",
                        dest="loss",
                        help="Type of loss (gaussian or xent)",
                        default='xent', type=str)

    parser.add_argument("-B", "--beta",
                        dest="beta",
                        help="Beta parameter for the VAE",
                        default=1.0, type=float)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default="./logs", type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=-1, type=int)


    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)

