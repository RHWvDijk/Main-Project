#tensorboard --logdir="C:\\Users\\s167917\\Documents\\#School\\Jaar 3\\3 Project Imaging\\GitHub\\code\\logs"
#http://localhost:6006
import os
#import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras import backend

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val','train')
     valid_path = os.path.join(base_dir, 'train+val','valid')

     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
train_gen, val_gen = get_pcam_generators('C:\\Users\s167917\\Documents\\#School\\Jaar 3\\3 Project Imaging\\data')

for i in range(20):
    input = Input(input_shape)
    pretrained = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    output = pretrained(input)
    output = GlobalAveragePooling2D()(output)
    output = Dropout(i/20)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(input, output)
    model.compile(SGD(lr=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

    # print a summary of the model on screen
    model.summary()

#    print(model.non_trainable_weights)
#    asd
#    # save the model and weights
    model_name = 'MNV2_dropout_test_{:0.2f}'.format(i/20)
    print(model_name)
    
    model_filepath = model_name + '.json'
    weights_filepath = model_name + '_weights.hdf5'

    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)

    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('logs', model_name))
    callbacks_list = [checkpoint, tensorboard]

    # train the model, note that we define "mini-epochs"
    train_steps = train_gen.n//train_gen.batch_size//20
    val_steps = val_gen.n//val_gen.batch_size//20

    # since the model is trained for only 10 "mini-epochs", i.e. half of the data is
    # not used during training
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=10,
                        callbacks=callbacks_list)
    backend.clear_session()