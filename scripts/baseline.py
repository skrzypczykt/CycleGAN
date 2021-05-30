import os
from gans.CycleGAN import load_data, CycleGAN, check_directories
import tensorflow as tf

if __name__ == '__main__':
    #data_dir = '/content/summer2winter-yosemite'
    data_dir = '../data/summer2winter_yosemite'

    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    RESTORE_TRAINING = False

    #PROJECT_ROOT_DIR = "/content/drive/MyDrive/CycleGAN/resnet/"
    PROJECT_ROOT_DIR = "../trainings/resnet/"

    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    checkpoint_path = os.path.join(PROJECT_ROOT_DIR, "checkpoints")

    check_directories(PROJECT_ROOT_DIR)

    #load data
    train_winter, test_winter, train_summer, test_summer, sample_A, sample_B = load_data(data_dir, IMG_WIDTH, IMG_HEIGHT)


    gan = CycleGAN(
        input_dim=(IMG_WIDTH, IMG_HEIGHT, 3)
        , learning_rate=0.0002
        , buffer_max_length=50
        , lambda_validation=1
        , lambda_reconstr=10
        , lambda_id=2
        , generator_type='resnet'
        , gen_n_filters=32
        , disc_n_filters=32
        , n_batches=961
    )

    mode = 'build'
    # mode = 'load'

    if mode == 'build':
        gan.save(checkpoint_path)
    else:
        #TODO: check
        max_epoch = 32 #???????
        gan.load_weights(os.path.join(checkpoint_path, f'weights/weights-{max_epoch}.h5'))


    EPOCHS = 1
    PRINT_EVERY_N_BATCHES = 1000

    """##train"""

    gan.train(tf.data.Dataset.zip((train_winter, train_summer))
              , run_folder=checkpoint_path
              , epochs=EPOCHS
              , test_A_file=sample_A
              , test_B_file=sample_B
              , batch_size=1
              , sample_interval=PRINT_EVERY_N_BATCHES)

    """##Loss"""

    fig = plt.figure(figsize=(20, 10))

    plt.plot([x[1] for x in gan.g_losses], color='green', linewidth=0.1)  # DISCRIM LOSS
    # plt.plot([x[2] for x in gan.g_losses], color='orange', linewidth=0.1)
    plt.plot([x[3] for x in gan.g_losses], color='blue', linewidth=0.1)  # CYCLE LOSS
    # plt.plot([x[4] for x in gan.g_losses], color='orange', linewidth=0.25)
    plt.plot([x[5] for x in gan.g_losses], color='red', linewidth=0.25)  # ID LOSS
    # plt.plot([x[6] for x in gan.g_losses], color='orange', linewidth=0.25)

    plt.plot([x[0] for x in gan.g_losses], color='black', linewidth=0.25)

    # plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

    plt.xlabel('batch', fontsize=18)
    plt.ylabel('loss', fontsize=16)

    plt.ylim(0, 5)

    plt.show()


    """#Test"""

    i = 0
    for imgA, img_B in tf.data.Dataset.zip((test_winter, test_summer)):
        gan.sample_images(imgA, img_B, i, os.path.join(PROJECT_ROOT_DIR, "test_results"), None, None, training=False)
        i = i + 1
