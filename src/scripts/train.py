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
from keras import backend as K
from keras.models import load_model
from collections import defaultdict
from math import ceil, floor
import pickle
import random

def main(dict_dir, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # Directories
    dictionary_dir = dict_dir
    val = '/dev_unseen/'
    speaker_list_train = os.listdir(dictionary_dir + '/train/')
    speaker_list_val = os.listdir(dictionary_dir + val)
    speaker_list_val_orig = speaker_list_val

    #speaker_list_val_controlled = speaker_list_val_orig*44
    #shuffle(speaker_list_val_controlled)
    
    # Training parameters
    training_params = {'num_epochs': 10,
                       'num_lstm_units': 32,
                       'num_FC_units': 32,
                       'lstm_dropout_rate': 0.2,
                       'FC_dropout_rate': 0.2,
                       'input_shape': (198, 39),
                       'alpha': 0.2,            #  alpha in paper
                       'BN_momentum': 0.99,
                       'l1_reg_weight': 10e-7}  


    # Data Generator parameters
    global params
    params = {'inp_dim': (198, 39),
              'target_dim': 1,
              'batch_size': 32,
              'num_speakers_per_miniepoch': 2,
              'shuffle': True}

    # Triplet Selection parameters go HERE

    # Model
    model = build(training_params)
    model.summary()

    opt = Adam(lr=0.001, decay=10) #RMSprop(lr=0.001) #Adam(lr=0.001)  # default lr=0.001
    model.compile(optimizer=opt,
                  loss=triplet_loss,
                  metrics=[triplet_loss])
    print("Model Compiled successfully")

    # Create Dictionaries of all Negative wavs for each speaker

    wavs_dict = defaultdict(list)
    negative_wavs = defaultdict(list)
    if (False):
    ## Train
        wavs_all = []
        speaker_list = speaker_list_train
        print('train')
        for sp_idx, speaker in enumerate(speaker_list):
            #print("Creating Dictionaries of wavs per speaker ",sp_idx)
            f = open("/home/rperi/exp_DAE/data/featsscp/clean/train/" + speaker + '/feats_merge.scp', 'r')
            # open("/home/raghuveer/tmp_data/DAE_exp/clean/" "/" + flag + "/" + speaker + '/wav.scp', 'r')
            # open("/home/rperi/exp_DAE/data/clean/" + "/" + flag + "/" + speaker + '/wav.scp', 'r')
            # open("/home/raghuveer/tmp_data/DAE_exp/clean/train/" + speaker + '/wav.scp', 'r')
            lines = f.readlines()
            wavs_dict[speaker] = [x.strip().split(" ")[0] for x in lines]
            wavs_all.append(list(wavs_dict[speaker]))
            f.close()

        wavs_all = [x for t in wavs_all for x in t]
        # print(wavs_all)
        for sp_idx, speaker in enumerate(speaker_list):
            # print(speaker)
            print("Creating Dictionaries of negative wavs per speaker ",sp_idx)
            negative_wavs[speaker] = [x for x in wavs_all if x not in list(wavs_dict[speaker])]

        ## Val
        wavs_all = []
        speaker_list = speaker_list_val
        for sp_idx, speaker in enumerate(speaker_list):
            #print("Creating Dictionaries of wavs per speaker ",sp_idx)
            f = open("/home/rperi/exp_DAE/data/featsscp/clean/dev_unseen/" + speaker + '/feats_merge.scp', 'r')
            # open("/home/raghuveer/tmp_data/DAE_exp/clean/" "/" + flag + "/" + speaker + '/wav.scp', 'r')
            # open("/home/rperi/exp_DAE/data/clean/" + "/" + flag + "/" + speaker + '/wav.scp', 'r')
            # open("/home/raghuveer/tmp_data/DAE_exp/clean/train/" + speaker + '/wav.scp', 'r')
            lines = f.readlines()
            wavs_dict[speaker] = [x.strip().split(" ")[0] for x in lines]
            wavs_all.append(list(wavs_dict[speaker]))
            f.close()

        wavs_all = [x for t in wavs_all for x in t]
        # print(wavs_all)
        for sp_idx,speaker in enumerate(speaker_list):
            print("Creating Dictionaries of negative wavs per speaker ",sp_idx)
            # print(speaker)
            negative_wavs[speaker] = [x for x in wavs_all if x not in list(wavs_dict[speaker])]

        del wavs_all
    ## Saving dictionaries for easy loading later
        f = open("wavs_dict.pickle",'wb')
        pickle.dump(wavs_dict, f)
        f.close()

        f = open("negative_wavs.pickle",'wb')
        pickle.dump(negative_wavs, f)
        f.close()
    
    else:
        f = open('wavs_dict.pickle', 'rb')
        wavs_dict = pickle.load(f)
        f.close()
        f = open('negative_wavs.pickle', 'rb')
        negative_wavs = pickle.load(f)
        f.close()
    shuffle(speaker_list_val)
    for epoch in range(training_params['num_epochs']):
        # Load the train and test data into generator

        #speaker_list_train = ['FEE005', 'FEE016']
        #speaker_list_val = ['FEE046', 'FEE087']
        print(speaker_list_train)

        shuffle(speaker_list_train)
        #shuffle(speaker_list_val)
        
        GeneratorObject = DataGenerator(**params)

        # Mini epochs
        num_mini_epochs = 30
        num_speakers_train_per_miniepoch = 5 #int(floor(len(speaker_list_train) / num_mini_epochs))
        num_speakers_val_per_miniepoch = 5 #int(floor(len(speaker_list_val) / num_mini_epochs))

        while num_speakers_val_per_miniepoch < 2:  # To deal with the case with too few speakers for validation
            speaker_list_val = [speaker_list_val, speaker_list_val]
            speaker_list_val = [x for t in speaker_list_val for x in t]
            num_speakers_val_per_miniepoch = int(floor(len(speaker_list_val) / num_mini_epochs))
        
        ############ SOME HARDCODED PARAMS ###########
        num_speakers_val_set = 5
        speaker_list_val_miniepoch = speaker_list_val[0:num_speakers_val_set]

        print("val list length = ",len(speaker_list_val_miniepoch))
        
        val_list_for_negatives = []
        for speaker in speaker_list_val_miniepoch:
            for i in range(780):
                val_list_for_negatives.append(random.choice([x for x in speaker_list_val_miniepoch if x != speaker]))

        #print(len(speaker_list_val_miniepoch), len(val_list_for_negatives))
        for mini_epoch in range(num_mini_epochs):
            print("mini epoch " + str(mini_epoch) + " / " + str(num_mini_epochs))
            start = time.time()
            if mini_epoch > 0 or epoch > 0:
                
                model_current = build(training_params)
                #model_current.summary()

                opt = Adam(lr=0.001)  # default lr=0.001
                model_current.compile(optimizer=opt,
                                    loss=triplet_loss,
                                    metrics=[triplet_loss])
                model_current.load_weights(model_dir + "/saved_model_latest.h5")
            else:
                model_current = model

            graph = tf.get_default_graph()
            speaker_list_train_miniepoch = speaker_list_train[mini_epoch*num_speakers_train_per_miniepoch:
                                                            (mini_epoch+1)*num_speakers_train_per_miniepoch]
            #if len(speaker_list_miniepoch)
            flag = "train"
            training_generator = GeneratorObject.load_DataGenerators(dictionary_dir, wavs_dict, negative_wavs,
                                                                     speaker_list_train_miniepoch, flag,
                                                                     model_current, graph, val_list_for_negatives)

#            speaker_list_val_miniepoch = speaker_list_val[mini_epoch * num_speakers_val_per_miniepoch:(mini_epoch + 1)*num_speakers_val_per_miniepoch]
            
            flag = val.strip("/")
            validation_generator = GeneratorObject.load_DataGenerators(dictionary_dir, wavs_dict, negative_wavs,
                                                                       speaker_list_val_miniepoch, flag,
                                                                       model_current, graph, val_list_for_negatives)

            train_steps = 110  # 3000 #floor(int(len(speaker_list_train)*780/params['batch_size']))  # 40C2 = 780
            # ~= num_speakers_train_per_miniepoch*780/params['batch_size']

            val_steps = 110 #45  # 300 #floor(int(len(speaker_list_val)*780/params['batch_size']))
            # ~= num_speakers_val_per_miniepoch*780/params['batch_size']

            #print(train_steps, val_steps)
            model_current = train(model_current, training_generator, validation_generator, epochs=1,
                  train_steps_per_epoch=train_steps,  # train_size/batch_size,
                  val_steps_per_epoch=val_steps)  # val_size/batch_size)

            print("Time for 1 mini epoch = " + str(time.time() - start) + " seconds")
            del training_generator, validation_generator
            model_current.save_weights(model_dir + "/saved_model_latest.h5")
            model_current.save_weights(model_dir + "/saved_model_miniepoch_" + str(mini_epoch) + "_epoch_" + str(epoch) +".h5")

            tf.reset_default_graph()
            K.clear_session()


if __name__ == '__main__':
    with tf.device('/gpu:2'):
        main(dict_dir, model_dir)
