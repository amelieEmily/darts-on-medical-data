import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torchvision.transforms as transforms
import logging
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import keras
config = tf.ConfigProto( device_count = {'GPU': 1 } )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

plt.rcParams['figure.figsize'] = (12,7)

# Input data files are available in the "../input/" directory.

import os
import sys

FILE_ABSOLUTE_PATH = os.path.abspath(__file__)
cell_images_folder_path = os.path.dirname(FILE_ABSOLUTE_PATH)
project_path = os.path.dirname(cell_images_folder_path)
src_path = os.path.join(project_path, 'RobustDARTS', 'src')
sys.path.append(src_path)
print(sys.path)
import utils


parser = argparse.ArgumentParser(description='baseline performances based on different traditional architecture.')
parser.add_argument('--network', type=str, default='custom1', help='type of network to run baseline on choose from: custom1, resnet18, xception')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('./baseline_logs',
                                      'log_{}.txt'.format(args.network)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


infected = os.listdir("./Parasitized")
infected_path = "./Parasitized"
uninfected = os.listdir("./Uninfected")
uninfected_path = "./Uninfected"

img_shape = (130,130,3)
image_gen = ImageDataGenerator(rotation_range = 20,
                              width_shift_range = 0.1,
                              height_shift_range=0.1,
                              rescale=1 / 255,
                              shear_range=0.1,
                              zoom_range=0.1,
                              horizontal_flip=True,
                              fill_mode='nearest',
                              validation_split=0.2)

train = image_gen.flow_from_directory('.',
                                     target_size =img_shape[:2],
                                     color_mode='rgb',
                                     batch_size = 16,
                                     class_mode='binary',shuffle=True,
                                     subset="training")

validation = image_gen.flow_from_directory('.',
                                     target_size = img_shape[:2],
                                     color_mode='rgb',
                                     batch_size = 16,
                                     class_mode='binary',
                                     subset="validation",shuffle=False)


if args.network == 'custom1':
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape = (130,130,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
elif args.network == 'resnet18':
    model = ResNet50(weights=None, input_shape=(130,130,3), classes=1)
elif args.network == 'xception':
    model = Xception(weights=None, input_shape=(130,130,3), classes=1)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


early = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit_generator(train,
                    epochs=20,
                    validation_data=validation,
                    callbacks=[early])

logging.info("param size = %fMB", utils.count_parameters_in_MB(model))