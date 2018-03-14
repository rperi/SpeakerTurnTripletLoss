import pickle
import numpy as np
import os
from keras import backend as K
from math import ceil
from collections import defaultdict
import random

def get_dictionaries(dict_dir, chunk):
    with open(dict_dir + '/input_dict_%d' % chunk + '.pickle', 'rb') as handle:
        input_dict = pickle.load(handle)

    return input_dict


def negative_sampling(speaker, chunk, anchor_positive_pairs, tmp_dictionary, negative_wav_ids, wavs_dict,
                      dict_dir, flag, graph, model_current, speaker_list):
    found_neg_FLAG = 0
    batch_size = 32
    num_batches = ceil(len(anchor_positive_pairs)/batch_size)
    triplets = []
    dict_dir_current = dict_dir + '/' + flag + '/'

    if flag == 'train':

        for batch in range(num_batches):
            iters = 0
            # print("For Speaker " + str(speaker) + " batch " + str(batch) + " / " + str(num_batches))
            anc_pos_batch = anchor_positive_pairs[batch * batch_size:(batch + 1) * batch_size]

            selected_idx = []
            pairs_batch = []
            pair_strings = []

            for idx, pairs in enumerate(anc_pos_batch):
                pairs_batch.append(pairs)
                pair_strings.append(str(pairs[0]) + "&" + str(pairs[1]))

            negative_speaker = defaultdict()
            negative_chunk = defaultdict()
            negative_string = defaultdict()

            while len(anc_pos_batch) >= 8 and iters < 10:
                iters += 1
                feats_anchor = np.zeros((len(anc_pos_batch),198,39))
                feats_positive = np.zeros((len(anc_pos_batch),198,39))
                feats_negative = np.zeros((len(anc_pos_batch),198,39))
                #print("Remaining elements in batch = " + str(len(anc_pos_batch)))
                for idx, pairs in enumerate(anc_pos_batch):
                    #print(speaker)
                    #pair_string = pair_strings[idx]
                    pair_string = str(pairs[0]) + "&" + str(pairs[1])
                    #print("Finding negatives for " + str(idx) + " / " + str(len(anc_pos_batch)))
                    feats_anchor[idx, :] = np.array(tmp_dictionary[pairs[0]]).reshape(1, 198, 39)
                    feats_positive[idx, :] = np.array(tmp_dictionary[pairs[1]]).reshape(1, 198, 39)
                    #print(feats_positive.shape)
                    for neg_idx, neg in enumerate(negative_wav_ids):
                        for sp in wavs_dict.keys():
                            if neg in list(wavs_dict[sp]):
                                negative_speaker = sp
                                break
                            else:
                                continue
                        dict_dir_negative_speaker = dict_dir_current + sp
                        #print(sp)

                        for neg_chunk in range(int(len(os.listdir(dict_dir_negative_speaker)))):
                            k = list((get_dictionaries(dict_dir_negative_speaker, neg_chunk)).keys())
                            if neg in k:
                                negative_chunk[pair_string] = neg_chunk
                                negative_string[pair_string] = neg
                                negative_dictionary = get_dictionaries(dict_dir_negative_speaker, neg_chunk)
                                k.clear()
                                break
                            else:
                                continue
                        #print(neg_chunk)
                        
                        break

                    feats_negative[idx, :] = np.array(negative_dictionary[neg]).reshape(1, 198, 39)
                    del negative_dictionary
                with graph.as_default():
                    get_output = K.function(
                        [model_current.get_layer('anchor').input, model_current.get_layer('positive').input,
                         model_current.get_layer('negative').input,
                         K.learning_phase()], [model_current.get_layer('DeltaPlusAlpha').output])
                    
                    output_batch = get_output([feats_anchor, feats_positive, feats_negative, 1])[0]
                    # learning phase = 0 for using model for evaluating rather than training

                    #output_batch = model_current.predict([feats_anchor, feats_positive, feats_negative])
                    #print((output_batch.shape))
                    output_batch = [x for t in output_batch for x in t]
                    #print(output_batch)
                    #del feats_anchor, feats_positive, feats_negative
                #print("Obtained output")
                #print(output_batch)
                #print(len(pairs_batch))
                #print(len(negative_string.keys()))
                #print(len(negative_chunk.keys()))
                for idx, out in enumerate(output_batch):
                    #print(out)
                    if out > 0:
                        selected_idx.append(idx)
                        pair = str(pairs_batch[idx][0]) + "&" + str(pairs_batch[idx][1])
                        triplets.append(str(speaker) + "#" + str(negative_speaker) + "#" +
                                        str(pairs_batch[idx][0]) + "#" + str(pairs_batch[idx][1]) + "#" +
                                        str(negative_string[pair]) + "#" + str(chunk) + "#" + str(negative_chunk[pair]) +
                                        '#' + str(out))

                not_selected_indices = [t for t in list(range(len(output_batch))) if t not in selected_idx]
                #print(not_selected_indices)
                if not_selected_indices != [None]:
                    anc_pos_batch = [pairs_batch[x] for x in not_selected_indices]
                else:
                    anc_pos_batch = []
            del feats_positive, feats_negative, feats_anchor
    else:
        for idx, pairs in enumerate(anchor_positive_pairs):
            rand_speaker = speaker_list[idx] # random.choice(speaker_list)
            dict_dir_current_speaker = dict_dir_current + str(rand_speaker)

            num_chunks = int(len(os.listdir(dict_dir_current_speaker)))
            rand_chunk = 0 # random.randint(0, num_chunks-1)

            neg_list = list((get_dictionaries(dict_dir_current_speaker, rand_chunk)).keys())

            rand_neg = neg_list[25]  # random.choice(neg_list)
            output = 'val'
            triplets.append(str(speaker) + "#" + str(rand_speaker) + "#" + \
                            str(pairs[0]) + "#" + str(pairs[1]) + "#" + \
                            str(rand_neg) + "#" + str(chunk) + "#" + str(rand_chunk) + \
                            "#" + str(output))
            #print(triplets)
    return triplets
