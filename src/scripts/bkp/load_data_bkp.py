import pickle
import numpy as np
import os
from keras import backend as K
from math import ceil


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
    
    for batch in range(num_batches):
    anc_pos_batch = anchor_positive_pairs[batch*batch_size:(batch+1)*batch_size]
    for idx, pairs in enumerate(anc_pos_batch):
        #print(idx)
        if flag == 'train':
            #print(idx)
            feats_anchor = np.array(tmp_dictionary[pairs[0]]).reshape(1, 198, 39)
            feats_positive = np.array(tmp_dictionary[pairs[1]]).reshape(1, 198, 39)

            for neg_idx, neg in enumerate(negative_wav_ids):
                for sp in wavs_dict.keys():
                    if neg in list(wavs_dict[sp]):
                        negative_speaker = sp
                        break
                    else:
                        continue
                dict_dir_negative_speaker = dict_dir_current + negative_speaker

                for neg_chunk in range(int(len(os.listdir(dict_dir_negative_speaker)))):
                    k = list((get_dictionaries(dict_dir_negative_speaker, neg_chunk)).keys())
                    if neg in k:
                        negative_dictionary = get_dictionaries(dict_dir_negative_speaker, neg_chunk)
                        k.clear()
                        break
                    else:
                        continue
                # print(neg_chunk)
                feats_negative = np.array(negative_dictionary[neg]).reshape(1, 198, 39)
                del negative_dictionary
                with graph.as_default():
                    get_output = K.function(
                        [model_current.get_layer('anchor').input, model_current.get_layer('positive').input,
                         model_current.get_layer('negative').input,
                         K.learning_phase()], [model_current.get_layer('DeltaPlusAlpha').output])

                    output = get_output([feats_anchor, feats_positive, feats_negative, 1])[0][0][0]
                del feats_negative
                # print("Obtained output")
                if output > 0.0:
                    found_neg_FLAG = 1
                    triplets.append(str(speaker) + "#" + str(negative_speaker) + "#" +
                                    str(pairs[0]) + "#" + str(pairs[1]) + "#" +
                                    str(neg) + "#" + str(chunk) + "#" + str(neg_chunk) +
                                    '#' + str(output))

                    break
                else:
                    continue
            del feats_anchor, feats_positive
        else:
            rand_speaker = random.choice(speaker_list)
            dict_dir_current_speaker = dict_dir_current + str(rand_speaker)

            num_chunks = int(len(os.listdir(dict_dir_current_speaker)))
            rand_chunk = random.randint(0, num_chunks-1)

            neg_list = list((get_dictionaries(dict_dir_current_speaker, rand_chunk)).keys())

            rand_neg = random.choice(neg_list)
            output = 'val'
            triplets.append(str(speaker) + "#" + str(rand_speaker) + "#" + \
                            str(pairs[0]) + "#" + str(pairs[1]) + "#" + \
                            str(rand_neg) + "#" + str(chunk) + "#" + str(rand_chunk) + \
                            "#" + str(output))
    return triplets
