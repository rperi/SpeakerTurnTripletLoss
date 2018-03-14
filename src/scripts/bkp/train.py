# -*- coding: utf-8 -*-

from scripts.DataGeneratorClass import DataGenerator
from scripts.model import build, train, NAME, triplet_loss
import os
from keras.optimizers import Adam, RMSprop
from keras import metrics
from random import shuffle
import tensorflow as tf
from math import floor
import time


def main(dict_dir, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # Directories
    dictionary_dir = dict_dir
    val = '/dev_unseen/'
    speaker_list_train = os.listdir(dictionary_dir + '/train/')
    speaker_list_val = os.listdir(dictionary_dir + val)

    # Training parameters
    training_params = {'num_epochs': 10,
                       'num_lstm_units': 16,
                       'num_FC_units': 16,
                       'lstm_dropout_rate': 0.1,
                       'FC_dropout_rate': 0.1,
                       'triplet_loss_margin': 0.1,
                       'input_shape': (198, 39),
                       'alpha': 0.2,            #  alpha in paper
                       'BN_momentum': 0.99}  


    # Data Generator parameters
    global params
    params = {'inp_dim': (198, 39),
              'target_dim': 1,
              'batch_size': 32,
              'num_speakers_per_miniepoch': 2,
              'shuffle': True}

    # Triplet Selection parameters
    model = build(training_params)
    model.summary()

    opt = RMSprop(lr=0.001) #Adam(lr=0.001)  # default lr=0.001
    model.compile(optimizer=opt,
                  loss=triplet_loss,
                  metrics=[triplet_loss])
    print("Model Compiled successfully")
    for epoch in range(training_params['num_epochs']):
        # Load the train and test data into generator
        start = time.time()

        if epoch > 0:
            load_model(model_dir + "/saved_model_latest.h5")

        graph = tf.get_default_graph()
        speaker_list_train = ['FEE005', 'FEE016']
        speaker_list_val = ['FEE046', 'FEE087']
        print(speaker_list_train)

        shuffle(speaker_list_train)
        shuffle(speaker_list_val)
        
        GeneratorObject = DataGenerator(**params)

        flag = "train"
        training_generator = GeneratorObject.load_DataGenerators(dictionary_dir, speaker_list_train,flag, model, graph, model_dir)

        flag = val.strip("/")
        validation_generator = GeneratorObject.load_DataGenerators(dictionary_dir, speaker_list_val, flag, model, graph, model_dir)
        train_steps = 3000 #floor(int(len(speaker_list_train)*780/params['batch_size']))  # 40C2 = 780
        val_steps = 300 #floor(int(len(speaker_list_val)*780/params['batch_size']))
        
        print(train_steps, val_steps)
        train(model, training_generator, validation_generator, epochs=1,
              train_steps_per_epoch=train_steps,  # train_size/batch_size,
              val_steps_per_epoch=val_steps)  # val_size/batch_size)

        print("Time for 1 epoch = " + str(time.time() - start) + " seconds")
        del training_generator, validation_generator
        model.save(model_dir + "/saved_model_epoch_" + str(epoch) + ".h5")


if __name__ == '__main__':
    with tf.device('/gpu:2'):
        main(dict_dir, model_dir)
