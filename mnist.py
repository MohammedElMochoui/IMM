"""
Simple MNIST sanity check for the VAE

"""

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, \
    Flatten, Dense, Input, Reshape, UpSampling2D, Conv2DTranspose
from keras import Model
import numpy as np
import util, random, tqdm, math

from argparse import ArgumentParser

from tensorboardX import SummaryWriter

def rec_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def go(options):

    tbw = SummaryWriter(log_dir=options.tb_dir)

    # Set random or det. seed
    if options.seed < 0:
        seed = random.randint(0, 1000000)
    else:
        seed = options.seed

    np.random.seed(seed)
    print('random seed: ', seed)

    # load data
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train[..., None]/255.0
    x_test  = x_test[..., None]/255.0

    n = x_train.shape[0]

    ### Build model
    ## Build the model

    input = Input(shape=(28, 28, 1))
    eps = Input(shape=(options.latent_size, ))

    a, b, c = 8, 32, 128

    h = Conv2D(a, (5, 5), activation='relu', padding='same')(input)
    h = Conv2D(a, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(a, (5, 5), activation='relu', padding='same')(h)
    h = MaxPooling2D((2, 2), padding='same')(h)

    h = Conv2D(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(b, (5, 5), activation='relu', padding='same')(h)
    h = MaxPooling2D((2, 2), padding='same')(h)

    h = Conv2D(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(c, (5, 5), activation='relu', padding='same')(h)
    h = MaxPooling2D((2, 2), padding='same')(h)

    h = Flatten()(h) # size is 4*4*c

    zmean = Dense(options.latent_size)(h)
    zlsig = Dense(options.latent_size)(h)

    kl = util.KLLayer()
    [zmean, zlsig] = kl([zmean, zlsig])
    zsample = util.Sample()([zmean, zlsig, eps])

    h = Dense(4 * 4 * c, activation='relu')(zsample)
    #  h = Dense(HEIGHT//(4*4*4) * WIDTH//(4*4*4) * 128, activation='relu')(zsample)

    h = Reshape((4, 4, 128))(h)

    h = Conv2DTranspose(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(c, (5, 5), activation='relu', padding='same')(h)
    h = UpSampling2D((3, 3))(h)

    h = Conv2DTranspose(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(b, (5, 5), activation='relu', padding='same')(h)
    h = UpSampling2D((2, 2))(h)

    h = Conv2DTranspose(a, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(a, (5, 5), activation='relu', padding='same')(h)
    output = Conv2DTranspose(1, (5, 5), activation='sigmoid')(h)

    encoder = Model(input, [zmean, zlsig])
    # decoder = Model(zsample, output)
    auto = Model([input, eps], output)

    opt = keras.optimizers.Adam(lr=options.lr)
    auto.compile(optimizer=opt,
                 loss=rec_loss)

    ### Fit model
    b = options.batch_size
    instances_seen = 0

    for e in range(options.epochs):
        for fr in tqdm.trange(0, 100, b):

            to = fr + b
            if to > n:
                to = n

            batch = x_train[fr:to, ...]
            bn = batch.shape[0]
            eps = np.random.randn(bn, options.latent_size)

            l = auto.train_on_batch([batch, eps], batch)

            instances_seen += batch.shape[0]

            if l.squeeze().ndim == 0:
                tbw.add_scalar('score/sum', float(l), instances_seen)
            else:
                tbw.add_scalar('score/sum', float(np.sum(l) / len(l)), instances_seen)

        ## Plot the latent space
        print('Plotting latent space.')

        latents = encoder.predict(x_test)[0]
        print('-- Computed latent vectors.')

        rng = np.max(latents[:, 0]) - np.min(latents[:, 0])

        print('-- L', latents[:10,:])
        print('-- range', rng)

        n_test = latents.shape[0]
        util.plot(latents, x_test, size=rng/math.sqrt(n_test), filename='mnist.{:04}.pdf'.format(e), invert=True)
        print('-- finished plot')


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=20, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="latent_size",
                        help="Size of the latent representation",
                        default=2, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/lm', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=1, type=int)


    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)