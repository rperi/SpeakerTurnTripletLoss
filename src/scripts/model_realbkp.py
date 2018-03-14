# -*- coding: utf-8 -*-
from keras import Input, metrics, regularizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import K, Dense, Dropout, LSTM, AveragePooling1D, Concatenate, Lambda, Subtract, Add
from keras.optimizers import Adam, Adadelta, SGD
import keras.backend as B
from keras.callbacks import Callback
import tensorflow as tf
from keras.layers import BatchNormalization as BN
import resource
NAME = "SequenceEmbedding"


def eucledian_dist(inputs):  # NOTE: Actually distance squared
    from keras import backend as B
    vec1, vec2 = inputs
    return Lambda(lambda x: B.sum(B.square(Subtract()([vec1, x])), axis=2))(vec2)


def triplet_dist(input_vecs):
    anc, pos, neg = input_vecs

    dist_anc_pos = Lambda(eucledian_dist)([anc, pos])
    dist_anc_neg = Lambda(eucledian_dist)([anc, neg])

    return Subtract()([dist_anc_pos, dist_anc_neg])


def triplet_loss(out_true,
                 out_pred):  # Essentially model gives the triplet loss as output, which needs to be positive
                             # and made as close to 0 as possible
    from keras import backend as B
    out = Lambda(lambda x: B.maximum(0.0, x))(out_pred)
    return Subtract()([out, out_true])


def create_network(shape, params):
    from keras import backend as B 
    num_lstm_units = params['num_lstm_units']
    lstm_dropout_rate = params['lstm_dropout_rate']
    num_FC_units = params['num_FC_units']
    FC_dropout_rate = params['FC_dropout_rate']
    BN_momentum = params['BN_momentum']
    
    input_layer = Input(shape=shape)
    
    # LSTM layers
    lstm_f = LSTM(num_lstm_units, activation='tanh', return_sequences=True)(input_layer)
    lstm_r = LSTM(num_lstm_units, activation='tanh', return_sequences=True, go_backwards=True)(input_layer)
    
    lstm_f = AveragePooling1D(pool_size=198)(lstm_f)
    lstm_r = AveragePooling1D(pool_size=198)(lstm_r)
    lstm_merge = Concatenate()([lstm_f, lstm_r])
    
    #lstm_merge = BN(momentum=BN_momentum)(lstm_merge)    
    # Fully Connected layers

    fc = Dense(num_FC_units, activation='tanh')(lstm_merge)
    fc = Dense(num_FC_units, activation='tanh')(fc)
    
    out = Lambda(lambda x: B.l2_normalize(x, axis=2))(fc)
    
    return Model(input_layer, out)


def build(params):
    alpha = params['alpha']
    input_shape = params['input_shape']
    network = create_network(input_shape, params)
    
    # Input layers
    inputs_anc = Input(shape=input_shape, name='anchor')
    inputs_pos = Input(shape=input_shape, name='positive')
    inputs_neg = Input(shape=input_shape, name='negative')
    
    # Output Layers
    out_anc = network(inputs_anc)
    out_pos = network(inputs_pos)
    out_neg = network(inputs_neg)
    
    # Loss being added as an additional layer

    delta = Lambda(lambda x: triplet_dist(x), name='Delta')([out_anc, out_pos, out_neg])
    loss = Lambda(lambda x: x + alpha, name='DeltaPlusAlpha')(delta)

    return Model([inputs_anc, inputs_pos, inputs_neg], loss, name=NAME)  # Since output is the loss. The target value = 0 for all samples


def train(model, train_generator, val_generator, epochs=100, train_steps_per_epoch=100, val_steps_per_epoch=100):

    #opt = Adam(lr=0.001)  # default lr=0.001
    #model.compile(optimizer=opt,
     #             loss=triplet_loss,
     #            metrics=[triplet_loss])

    history = LossHistory()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks=[history])
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        #self.file = open("/home/rperi/exp_DAE/SpeakerTurnLossHistory/train_loss.txt", 'a')
        #open("/home/raghuveer/tmp_data/DAE_exp/loss_history/loss.txt", 'a')
    
    def on_batch_begin(self, batch, logs={}):
        self.file = open("/home/rperi/exp_DAE/SpeakerTurnLossHistory/train_loss.txt", 'a')
    def on_batch_end(self, batch, logs={}):
        self.file.writelines(str(logs.get('loss')))
        self.file.writelines("\n")
        self.losses.append(logs.get('loss'))
        self.file.close()
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def on_train_end(self, logs={}):
        self.file = open("/home/rperi/exp_DAE/SpeakerTurnLossHistory/train_loss.txt", 'a')
        self.file.writelines("End of epoch.. Beginning new epoch")
        self.file.close()


def train_old(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
    model.summary()
    model.compile(optimizer=Adadelta(lr=1.0, decay=0.2),
                  loss=K.binary_crossentropy,
                  metrics=[metrics.binary_accuracy, metrics.mean_squared_error])

    model.fit(x=x_train, y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/{}".format(NAME),
                  write_images=True,
                  histogram_freq=5,
                  batch_size=batch_size
              ), ModelCheckpoint(
                  "models/" + NAME + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",
                  monitor='val_loss',
                  verbose=1,
                  save_best_only=True,
                  mode='auto'
              )])
