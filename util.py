import keras

from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Layer, Input
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
import keras.utils
import keras.backend as K

import numpy as np
import os, sys
import datetime, pathlib
from keras.preprocessing import sequence

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


import cv2

def to_categorical(batch, num_classes):
    """
    Converts a batch of length-padded integer sequences to a one-hot encoded sequence
    :param batch:
    :param num_classes:
    :return:
    """

    b, l = batch.shape

    out = np.zeros((b, l, num_classes))

    for i in range(b):
        seq = batch[0, :]
        out[i, :, :] = keras.utils.to_categorical(seq, num_classes=num_classes)

    return out

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class KLLayer(Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.

    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, weight = None, *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        if self.weight is None:
            loss = kl_batch
        else
            loss = kl_batch * self.weight

        self.add_loss(loss)

        return inputs

class Sample(Layer):

    """
    Performs sampling step

    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, eps = inputs

        eps = Input(tensor=K.random_normal(shape=K.shape(mu) ))

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _, _ = input_shape
        return shape_mu


def loadmovie(file):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return buf

def ensure(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def plot(latents, images, size=0.00001, filename='latent_space.pdf', invert=False):

    assert(latents.shape[0] == images.shape[0])

    mn, mx = np.min(latents), np.max(latents)

    n, h, w, c = images.shape

    aspect = h/w

    fig = plt.figure(figsize=(64,64))
    ax = fig.add_subplot(111)

    for i in range(n):
        x, y = latents[i, 0:2]

        im = images[i, :]

        ax.imshow(im if c > 1 else im.squeeze(2), extent=(x, x + size, y, y + size*aspect), cmap='gray_r' if invert else 'gray')

    # ax.scatter(latents[:, 0], latents[:, 1], alpha=0.01, linewidth=0)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    plt.savefig(filename)
    plt.close(fig)