from glob import glob
from collections import defaultdict
import pickle
from math import ceil
import random
import os
from itertools import combinations as C
from keras import backend as K
from scripts.model import triplet_dist
import numpy as np
import time
from scripts.load_data import get_dictionaries, negative_sampling
from keras.models import load_model

class DataGenerator(object):
    def __init__(self, inp_dim=(198, 39), target_dim=1, batch_size=32, num_speakers_per_miniepoch=1, shuffle=True):
        """initialization"""
        self.inp_dim = inp_dim
        self.target_dim = target_dim

        self.batch_size = batch_size
        self.num_speakers_per_miniepoch = num_speakers_per_miniepoch
        self.shuffle = shuffle

    def load_DataGenerators(self, dictionary_dir, speaker_list, flag, model_current, graph, model_dir):

        while True:
            print(flag)
            num_speakers_per_miniepoch = int(self.num_speakers_per_miniepoch)
            print("num_speakers = ", len(speaker_list))
            num_miniepochs = int(ceil(len(speaker_list)/num_speakers_per_miniepoch))

            error_flag = 0
            wavs_dict = defaultdict()
            wavs_all = []
            negative_wavs = defaultdict()
            for speaker in speaker_list:
                f = open("/home/rperi/exp_DAE/data/featsscp/clean" + "/" + flag + "/" + speaker + '/feats_merge.scp', 'r')
# open("/home/raghuveer/tmp_data/DAE_exp/clean/" "/" + flag + "/" + speaker + '/wav.scp', 'r')
                # open("/home/rperi/exp_DAE/data/clean/" + "/" + flag + "/" + speaker + '/wav.scp', 'r')
                # open("/home/raghuveer/tmp_data/DAE_exp/clean/train/" + speaker + '/wav.scp', 'r')
                lines = f.readlines()
                wavs_dict[speaker] = [x.strip().split(" ")[0] for x in lines]
                wavs_all.append(list(wavs_dict[speaker]))
                f.close()

            wavs_all = [x for t in wavs_all for x in t]
            #print(wavs_all)
            for speaker in speaker_list:
                #print(speaker)
                negative_wavs[speaker] = [x for x in wavs_all if x not in list(wavs_dict[speaker])]
            print("num_mini_epochs=",num_miniepochs)
            for mini_epoch in range(num_miniepochs):
                #if (mini_epoch + 1) % 2 == 0:
                #K.clear_session()
                #model_current = load_model(model_dir + "/saved_model_latest.h5")
                #graph = tf.get_default_graph()
                triplets = []
                triplets_miniepoch = []
                speaker_list_current = speaker_list[mini_epoch*num_speakers_per_miniepoch: (mini_epoch+1)*num_speakers_per_miniepoch]
                st = time.time()
                print("mini epoch " + str(mini_epoch) + " / " + str(num_miniepochs))
                print("Current speaker list = ", speaker_list_current)
                for speaker in speaker_list_current:
                    print(speaker)
                    #start = time.time()
                    dict_dir_current = dictionary_dir + '/' + flag + '/'

                    dict_dir_current_speaker = dict_dir_current + str(speaker)
                    print(dict_dir_current_speaker)
                    num_chunks = int(len(os.listdir(dict_dir_current_speaker)))
                    chunk = random.randint(0, num_chunks-1)
                    #print("chunk",chunk)
                    tmp_dictionary = get_dictionaries(dict_dir_current_speaker, chunk)
                    #print(len(tmp_dictionary.keys()))
                    anchor_positive_pairs = list(C(tmp_dictionary.keys(), 2))
                    
                    negative_wav_ids = list(negative_wavs[speaker])
                    random.shuffle(negative_wav_ids)
                    #print((anchor_positive_pairs))

                    tmp = negative_sampling(speaker, chunk, anchor_positive_pairs, tmp_dictionary, negative_wav_ids, wavs_dict, dictionary_dir, flag, graph, model_current, speaker_list)
                    print("Done negative sampling for speaker ",speaker)
                    del tmp_dictionary
                    triplets_miniepoch.append(tmp)

                    #print(time.time() - start, "seconds")
                #print(triplets_miniepoch)
                print("Found triplets for mini epoch", mini_epoch)
                triplets = [t for p in triplets_miniepoch for t in p]
                random.shuffle(triplets)
                if flag == 'train':
                    f = open("/home/rperi/exp_DAE/SpeakerTurnLossHistory/triplets.txt", 'a')
                    f.writelines(str(triplets) + "\n")
                    f.close()
                num_triplets = len(triplets)
                print("Num Triplets = ",num_triplets)
                batch_size = int(self.batch_size)
                num_batches = int(num_triplets/batch_size)
                print("num batches",num_batches)
                for b in range(num_batches):
                    current_batch = list(triplets[b*batch_size:(b+1)*batch_size])

                    anc, pos, neg, tar = self.__data_generation(dictionary_dir, flag, current_batch)

                    yield [anc, pos, neg], tar
                del triplets, triplets_miniepoch
                print("Time fof mini epoch = ", str(time.time() - st))
                model_current.save(model_dir + "/saved_model_latest.h5")
            del negative_wavs, wavs_dict


    def __data_generation(self, dict_dir, flag, batch):
        anchor_feats = np.empty((self.batch_size, self.inp_dim[0], self.inp_dim[1]))
        positive_feats = np.empty((self.batch_size, self.inp_dim[0], self.inp_dim[1]))
        negative_feats = np.empty((self.batch_size, self.inp_dim[0], self.inp_dim[1]))
        target_feats = np.empty((self.batch_size, self.target_dim))

        dict_dir_current = dict_dir + '/' + flag + '/'

        for ind, utt in enumerate(batch):
            speaker = utt.strip().split("#")[0]
            neg_speaker = utt.strip().split("#")[1]
            anchor = utt.strip().split("#")[2]
            positive = utt.strip().split("#")[3]
            negative = utt.strip().split("#")[4]
            ch = int(utt.strip().split("#")[5])
            neg_ch = int(utt.strip().split("#")[6])

            dict_pos = get_dictionaries(dict_dir_current + str(speaker), ch)
            anchor_feats[ind, :] = dict_pos[anchor]
            positive_feats[ind, :] = dict_pos[positive]

            dict_neg = get_dictionaries(dict_dir_current + str(neg_speaker), neg_ch)
            negative_feats[ind, :] = dict_neg[negative]

            target_feats[ind, :] = float(0)
            
            del dict_pos, dict_neg

        return anchor_feats, positive_feats, negative_feats, target_feats


