#!/usr/bin/python
 # -*- coding: utf-8 -*-
 
from hyperopt import hp

from keras.models import Model
from keras.layers import Input, Flatten
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

        print('>>> Testing parameters: Dense1({}), Dense2({})'.format(p['dense01'], p['dense02']))

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
            'dense01' : hp.choice('dense01', [64, 128, 256, 512]),
            'dense02' : hp.choice('dense02', [0, 64, 128, 256]),
        }

    def define_architecture(self, in_shape, hp_params):
        print('Testing parameters')
        print(hp_params)

        p = hp_params
        # Input
        in_layer =  Input(shape=in_shape)

        # Filters
        _filters = 100
        _kernel_size = 3
        _pool_size = 2
        conv = Conv1D(filters=_filters, kernel_size=_kernel_size, activation='relu', name='conv1')(in_layer)
        if _pool_size > 0:
            conv = MaxPooling1D(pool_size=_pool_size, name='pool1')(conv)

        # Flat
        flat = Flatten()(conv)

        # Dropout
        drop = Dropout(.1)(flat)

        # Fully connect
        dense1 = Dense(p['dense1'], activation='relu')(drop)

        # Output
        out_layer = Dense(1, activation='sigmoid')(dense1)

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