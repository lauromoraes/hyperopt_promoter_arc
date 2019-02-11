#!/usr/bin/python
 # -*- coding: utf-8 -*-
 
from hyperopt import hp

from keras.models import Model
from keras.layers import Input, Flatten, Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D

class Architecture(object):
    def __init__(self, input_data):
        self.input_data = input_data
        self.define_space()

    def get_space(self):
        return self.space

    def define_space(self):
        raise NotImplementedError("Please Implement this method")

    def define_architecture(self, in_shape, hp_params):
        raise NotImplementedError("Please Implement this method")

# =================================================================

class MLPArchitecture(Architecture):
    def __init__(self, input_data):
        super(MLPArchitecture, self).__init__(input_data)
        
    def define_space(self):
        self.space = {
            'dense01' : hp.choice('dense01', [64, 128, 256, 512]),
            'dense02' : hp.choice('dense02', [0, 64, 128, 256]),
        }

    def define_architecture(self, in_shape, hp_params):
        p = hp_params

        # Input
        in_layer =  Input(shape=in_shape)


        # Fully connect 1
        dense = Dense(p['dense01'], name='dense1', activation='relu')(in_layer)

        if p['dense02'] != 0:
            # Fully connect 2
            dense = Dense(p['dense02'], name='dense2', activation='relu')(dense)

        # Dropout
        drop = Dropout(.1)(dense)

        # Output
        out_layer = Dense(1, name='output', activation='sigmoid')(dense)

        return in_layer, out_layer

# =================================================================

class ConvArchitecture(Architecture):
    def __init__(self, input_data):
        super(ConvArchitecture, self).__init__(input_data)
        
    def define_space(self):
        self.space = {
            'conv01_filters' : hp.choice('conv01_filters', [128, 256]),
            'conv01_ksize'   : hp.choice('conv01_ksize',   [3, 9, 11]),
            'conv01_psize'   : hp.choice('conv01_psize',   [2, 3]),
            'dense01' : hp.choice('dense01', [256, 512]),
            'dense02' : hp.choice('dense02', [64, 128]),
        }

    def define_architecture(self, in_shape, hp_params):

        print(in_shape)

        p = hp_params
        # Input
        in_layer =  Input(shape=in_shape)

        _embedding_dims = 5
        _maxlen = in_shape[0]
        emb = Embedding(4, _embedding_dims, input_length=_maxlen)(in_layer)

        # Filters
        _filters = p['conv01_filters']
        _kernel_size = p['conv01_ksize']
        _pool_size = p['conv01_psize']
        _pool_stride = 2

        conv = Conv1D(filters=_filters, kernel_size=_kernel_size, activation='relu', name='conv1')(emb)
        if _pool_size > 0:
            conv = MaxPooling1D(pool_size=_pool_size, strides=_pool_stride,  name='pool1')(conv)

        # Flat
        flat = Flatten()(conv)

        # Dropout
        drop = Dropout(.1)(flat)

        # Fully connect
        dense = Dense(p['dense01'], activation='relu')(drop)
        dense = Dense(p['dense02'], activation='relu')(dense)

        # Output
        out_layer = Dense(1, activation='sigmoid')(dense)

        return in_layer, out_layer

# =================================================================

class ArchitectureFactory(object):
    def __init__(self, input_data):
        self.input_data = input_data

    def get_architecture(self, architecture_type):
        input_shape = self.input_data.shape[1]
        architecture = None
        if architecture_type == 'MLP':
            architecture = MLPArchitecture(input_shape)
        elif architecture_type == 'Conv':
            architecture = ConvArchitecture(input_shape)
        else:
            print('ERROR: "{}" is not a valid architecture type.'.format(architecture_type))
        return architecture