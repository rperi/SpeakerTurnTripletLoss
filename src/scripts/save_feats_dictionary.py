# This code uses kaldi-io-for-python library to read the kaldi mfcc files and saves them as dictionaries
# indexed by wav ids into disk. The saved dictionary can be loaded by batch generator during DAE training

# import sys
# sys.path.append('/home/raghuveer/tools/kaldi-io-for-python/')

import os
import kaldi_io
import pickle
from collections import defaultdict
import time
import numpy as np


def save(feature_directory, dictionary_directory_out):

    #num_chunks = 10  # number of chunks of chopped scp file

    feats_file_dir = feature_directory
    out_dict_dir = dictionary_directory_out

    split = 'dev_seen/'
    print(split)

    # Input dictionary
    input_dict = defaultdict()
    speakers = os.listdir(feature_directory + split)

    for speaker in speakers:
        num_chunks = int(len(os.listdir(feats_file_dir + split + speaker)))

        for chunk in range(num_chunks):
            start_time = time.time()
            c = chunk+1
            chunk_num = '%03d' % c
            feats_file_chunk = feats_file_dir + split + speaker + '/feats.' + chunk_num + '.scp'

            #if wav_id not in input_dict_train.keys() or wav_id not in input_dict_val.keys():
            if split == 'train/':
                input_dict = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_chunk)}

            if not os.path.exists(out_dict_dir + split + speaker):
                os.makedirs(out_dict_dir + split + speaker)

            with open(out_dict_dir + split + speaker + '/input_dict_' + str(chunk) + '.pickle', 'wb') as d:
                pickle.dump(input_dict, d, protocol=pickle.HIGHEST_PROTOCOL)
            input_dict.clear()

            print(time.time() - start_time)

        print('Success.. Saved features as dictionary into ' + out_dict_dir + split + speaker)


if __name__ == '__main__':
    save(feature_directory, dictionary_directory_out)
