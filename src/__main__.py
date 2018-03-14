# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('/home/rperi/tools/kaldi-io-for-python/')
print(sys.path)

os.environ['KERAS_BACKEND']='tensorflow'
from scripts.save_feats_dictionary import save
from scripts.train import main

if __name__ == '__main__':

    save_Dict = 0  # Toggle to save dictionary from features. Needs to be done atleast once

    feature_directory = '/home/raghuveer/tmp_data/DAE_exp/data/concat/'
    dictionary_directory_out = '/home/rperi/exp_DAE/data/feats_dictionaries_speakerseg/'
    # '/home/rperi/exp_DAE/data/feats_dictionaries_speakerseg/'
    # '/home/raghuveer/tmp_data/DAE_exp/feats_dictionaries_speakerseg/'
    model_dir = '/home/rperi/exp_DAE/SpeakerTurnSavedModels/saved_models/'
    # '/home/raghuveer/tmp_data/DAE_exp/saved_models/'
    # '/home/rperi/exp_DAE/SpeakerTurn/saved_models/'
    if save_Dict == 1:
        save(feature_directory, dictionary_directory_out)
    else:
        main(dictionary_directory_out, model_dir)
