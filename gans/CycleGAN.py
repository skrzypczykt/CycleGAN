"""Baseline cycleGAN2-256.ipynb

Original file is located at
    https://colab.research.google.com/drive/1XtFW9RaIAI8AqWMnrIX-rDnnax570HYr
"""

# Download data

# !wget https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be66e78_summer2winter-yosemite/summer2winter-yosemite.zip
# !unzip -q summer2winter-yosemite.zip
# !rm summer2winter-yosemite.zip
# !mv ./summer2winter_yosemite ./summer2winter-yosemite


# source summer2winter_yosemite
# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# !pip install git+https://www.github.com/keras-team/keras-contrib.git
# !pip install -U tensorflow-addons
# %load_ext autoreload
# %autoreload 2


import datetime
import os
import pickle as pkl
import random
import time
from collections import deque
import re

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import tensorflow_addons
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow_addons.layers import InstanceNormalization

# from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 1

BUFFER_SIZE = 1000
BATCH_SIZE = 1


def load_data(data_dir, IMG_WIDTH, IMG_HEIGHT):
    winter_train = tf.keras.preprocessing.image_dataset_from_directory(os.path.join(data_dir, "winter"),
                                                                       image_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                       batch_size=BATCH_SIZE)
    summer_train = tf.keras.preprocessing.image_dataset_from_directory(os.path.join(data_dir, "summer"),
                                                                       image_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                       batch_size=BATCH_SIZE)

    winter_test = tf.keras.preprocessing.image_dataset_from_directory(os.path.join(data_dir, "test_winter"),
                                                                      image_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                      batch_size=BATCH_SIZE)
    summer_test = tf.keras.preprocessing.image_dataset_from_directory(os.path.join(data_dir, "test_summer"),
                                                                      image_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                      batch_size=BATCH_SIZE)
    train_winter = winter_train.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).prefetch(AUTOTUNE)

    train_summer = summer_train.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).prefetch(AUTOTUNE)

    test_winter = winter_test.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).prefetch(AUTOTUNE)

    test_summer = summer_test.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).prefetch(AUTOTUNE)

    # samples
    sample_A = next(iter(test_winter))
    sample_B = next(iter(test_summer))

    #data len
    len_A = len(list(winter_train))
    len_B = len(list(summer_train))

    return train_winter, test_winter, train_summer, test_summer, sample_A, sample_B, min(len_A, len_B)


# !mkdir
# {PROJECT_ROOT_DIR}
# !mkdir
# {IMAGES_PATH}
# !mkdir
# {checkpoint_path}

def plot_gan_results(gan):

  fig = plt.figure(figsize=(20, 10))

  plt.plot([x[1] for x in gan.g_losses], color='green', linewidth=0.1)  # DISCRIM LOSS
  # plt.plot([x[2] for x in gan.g_losses], color='orange', linewidth=0.1)
  plt.plot([x[3] for x in gan.g_losses], color='blue', linewidth=0.1)  # CYCLE LOSS
  # plt.plot([x[4] for x in gan.g_losses], color='orange', linewidth=0.25)
  plt.plot([x[5] for x in gan.g_losses], color='red', linewidth=0.25)  # ID LOSS
  # plt.plot([x[6] for x in gan.g_losses], color='orange', linewidth=0.25)

  plt.plot([x[0] for x in gan.g_losses], color='black', linewidth=0.25)


  plt.xlabel('batch', fontsize=18)
  plt.ylabel('loss', fontsize=16)

  plt.ylim(0, 5)

  plt.show()


def generate_test_results(gan, datasets, PROJECT_ROOT_DIR):
  i = 0
  for imgA, img_B in datasets:
      gan.sample_images(imgA, img_B, i, os.path.join(PROJECT_ROOT_DIR, "test_results"), None, None, training=False)
      i = i + 1


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image_train(image, label):
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image)
    return image



def get_last_epoch(checkpoint_path):
    dir = os.path.join(checkpoint_path, 'weights')
    #/weights-{max_epoch}.h5')
    checkpoints = [re.search('(?<=weights-)\d+(?=.h5)',file_name) for file_name in os.listdir(dir) ]
    checkpoints = [int(reg.group(0)) for reg in checkpoints if reg]
    return max(checkpoints)

"""#CycleGAN definition"""
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config


# Commented out IPython magic to ensure Python compatibility.
class CycleGAN():
    def __init__(self
                 , input_dim
                 , learning_rate
                 , lambda_validation
                 , lambda_reconstr
                 , lambda_id
                 , generator_type
                 , gen_n_filters
                 , disc_n_filters
                 , n_batches
                 , buffer_max_length=50
                 ):

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.buffer_max_length = buffer_max_length
        self.lambda_validation = lambda_validation
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id
        self.generator_type = generator_type
        self.gen_n_filters = gen_n_filters
        self.n_batches = n_batches
        self.disc_n_filters = disc_n_filters

        # Input shape
        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self.buffer_A = deque(maxlen=self.buffer_max_length)
        self.buffer_B = deque(maxlen=self.buffer_max_length)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 3)
        self.disc_patch = (patch, patch, 1)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.compile_models()

    def compile_models(self):

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        self.d_A.compile(loss='mse',
                         optimizer=Adam(self.learning_rate, 0.5),
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=Adam(self.learning_rate, 0.5),
                         metrics=['accuracy'])

        # Build the generators
        if self.generator_type == 'unet':
            self.g_AB = self.build_generator_unet()
            self.g_BA = self.build_generator_unet()
        else:
            self.g_AB = self.build_generator_resnet()
            self.g_BA = self.build_generator_resnet()

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[self.lambda_validation, self.lambda_validation,
                                            self.lambda_reconstr, self.lambda_reconstr,
                                            self.lambda_id, self.lambda_id],
                              optimizer=Adam(0.0002, 0.5))

        self.d_A.trainable = True
        self.d_B.trainable = True

    def build_generator_unet(self):

        def downsample(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation('relu')(d)

            return d

        def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = Activation('relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)

            u = Concatenate()([u, skip_input])
            return u

        # Image input
        img = Input(shape=self.img_shape)

        # Downsampling
        d1 = downsample(img, self.gen_n_filters)
        d2 = downsample(d1, self.gen_n_filters * 2)
        d3 = downsample(d2, self.gen_n_filters * 4)
        d4 = downsample(d3, self.gen_n_filters * 8)
        d5 = downsample(d4, self.gen_n_filters * 16)

        # Upsampling

        u1 = upsample(d5, d4, self.gen_n_filters * 8)
        u2 = upsample(u1, d3, self.gen_n_filters * 4)
        u3 = upsample(u2, d2, self.gen_n_filters * 2)
        u4 = upsample(u3, d1, self.gen_n_filters)

        u5 = UpSampling2D(size=2)(u4)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)

        return Model(img, output_img)

    def build_generator_resnet(self):

        def conv7s1(layer_input, filters, final):
            y = ReflectionPadding2D(padding=(3, 3))(layer_input)
            y = Conv2D(filters, kernel_size=(7, 7), strides=1, padding='valid', kernel_initializer=self.weight_init)(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
                y = Activation('relu')(y)
            return y

        def downsample(layer_input, filters):
            y = Conv2D(filters, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=self.weight_init)(
                layer_input)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)
            return y

        def residual(layer_input, filters):
            shortcut = layer_input
            y = ReflectionPadding2D(padding=(1, 1))(layer_input)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer=self.weight_init)(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)

            y = ReflectionPadding2D(padding=(1, 1))(y)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer=self.weight_init)(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

            return Add()([shortcut, y])

        def upsample(layer_input, filters):
            y = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same',
                                kernel_initializer=self.weight_init)(layer_input)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)

            return y

        # Image input
        img = Input(shape=self.img_shape)

        y = img

        y = conv7s1(y, self.gen_n_filters, False)
        y = downsample(y, self.gen_n_filters * 2)
        y = downsample(y, self.gen_n_filters * 2)
        y = downsample(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = residual(y, self.gen_n_filters * 4)
        y = upsample(y, self.gen_n_filters * 2)
        y = upsample(y, self.gen_n_filters * 2)
        y = upsample(y, self.gen_n_filters)
        y = conv7s1(y, 3, True)
        output = y

        return Model(img, output)

    def build_discriminator(self):

        def conv4(layer_input, filters, stride=2, norm=True):
            y = Conv2D(filters, kernel_size=(4, 4), strides=stride, padding='same',
                       kernel_initializer=self.weight_init)(layer_input)

            if norm:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

            y = LeakyReLU(0.2)(y)

            return y

        img = Input(shape=self.img_shape)

        y = conv4(img, self.disc_n_filters, stride=2, norm=False)
        y = conv4(y, self.disc_n_filters * 2, stride=2)
        y = conv4(y, self.disc_n_filters * 4, stride=2)
        y = conv4(y, self.disc_n_filters * 8, stride=1)

        output = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=self.weight_init)(y)

        return Model(img, output)

    def train_discriminators(self, imgs_A, imgs_B, valid, fake):

        # Translate images to opposite domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        self.buffer_B.append(fake_B)
        self.buffer_A.append(fake_A)

        fake_A_rnd = random.sample(self.buffer_A, min(len(self.buffer_A), len(imgs_A)))
        fake_B_rnd = random.sample(self.buffer_B, min(len(self.buffer_B), len(imgs_B)))

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )

    def train_generators(self, imgs_A, imgs_B, valid):

        # Train the generators
        return self.combined.train_on_batch([imgs_A, imgs_B],
                                            [valid, valid,
                                             imgs_A, imgs_B,
                                             imgs_A, imgs_B])

    def train(self, data_loader, run_folder, epochs, test_A_file, test_B_file, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch + 1, self.epoch + epochs + 1 ):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader):

                d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
                g_loss = self.train_generators(imgs_A, imgs_B, valid)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print(
                    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}, acc: {}] [G loss: {}, adv: {}, recon:{}, id: ] time: {} ".format(
                    epoch, self.epoch + epochs + 1,
                    batch_i, self.n_batches,
                    d_loss[0], 100 * d_loss[7],
                    g_loss[0],
                    np.sum(g_loss[1:3]),
                    np.sum(g_loss[3:5]),
                    np.sum(g_loss[5:7]),
                    elapsed_time))

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(imgs_A, imgs_B, batch_i, run_folder, test_A_file, test_B_file)
                self.combined.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.combined.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

        self.epoch += epochs

    def sample_images(self, train_im_A, train_im_B, batch_i, run_folder, test_A_file, test_B_file, training=True):

        r, c = 2, 4

        for p in range(1 + training):

            if p == 0:
                imgs_A = train_im_A
                imgs_B = train_im_B
            else:
                imgs_A = test_A_file
                imgs_B = test_B_file

            # Translate images to the other domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)
            # Translate back to original domain
            reconstr_A = self.g_BA.predict(fake_B)
            reconstr_B = self.g_AB.predict(fake_A)

            # ID the images
            id_A = self.g_BA.predict(imgs_A)
            id_B = self.g_AB.predict(imgs_B)

            gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = np.clip(gen_imgs, 0, 1)

            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25, 12.5))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(run_folder, "images/%d_%d_%d.png" % (p, self.epoch, batch_i)))
            plt.close()

    def plot_model(self, run_folder):
        plot_model(self.combined, to_file=os.path.join(run_folder, 'viz/combined.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.d_A, to_file=os.path.join(run_folder, 'viz/d_A.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.d_B, to_file=os.path.join(run_folder, 'viz/d_B.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.g_BA, to_file=os.path.join(run_folder, 'viz/g_BA.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.g_AB, to_file=os.path.join(run_folder, 'viz/g_AB.png'), show_shapes=True, show_layer_names=True)

    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                , self.learning_rate
                , self.buffer_max_length
                , self.lambda_validation
                , self.lambda_reconstr
                , self.lambda_id
                , self.generator_type
                , self.gen_n_filters
                , self.disc_n_filters
            ], f)
        #TODO: fix
        #self.plot_model(folder)

    def save_model(self, run_folder):

        self.combined.save(os.path.join(run_folder, 'model.h5'))
        self.d_A.save(os.path.join(run_folder, 'd_A.h5'))
        self.d_B.save(os.path.join(run_folder, 'd_B.h5'))
        self.g_BA.save(os.path.join(run_folder, 'g_BA.h5'))
        self.g_AB.save(os.path.join(run_folder, 'g_AB.h5'))

        # pkl.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_weights(self, filepath):
        self.combined.load_weights(filepath)


def check_directories(PROJECT_ROOT_DIR):
    os.makedirs(PROJECT_ROOT_DIR, exist_ok = True )

    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "images"), exist_ok = True )

    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "checkpoints"), exist_ok = True)

    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "checkpoints", "viz"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "checkpoints", "weights"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "checkpoints", "images"), exist_ok=True)

    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "test_results"), exist_ok = True)
    os.makedirs(os.path.join(PROJECT_ROOT_DIR, "test_results", "images"), exist_ok=True)



"""#Data Loader Definition"""

"""#Main"""



# !mkdir
# {PROJECT_ROOT_DIR}
# !mkdir
# {os.path.join(PROJECT_ROOT_DIR, "checkpoints")}
# !mkdir
# {os.path.join(PROJECT_ROOT_DIR, "checkpoints", "viz")}
# !mkdir
# {os.path.join(PROJECT_ROOT_DIR, "checkpoints", "images")}
# !mkdir
# {os.path.join(PROJECT_ROOT_DIR, "checkpoints", "weights")}


#
#
# plot_model(gan.g_BA)
#
# gan.g_AB.summary()
#
# gan.d_A.summary()
#
# gan.d_B.summary()
#
