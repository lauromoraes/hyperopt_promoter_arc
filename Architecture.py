#!/usr/bin/python
 # -*- coding: utf-8 -*-
 
from hyperopt import hp

from keras.models import Model
from keras.layers import Input, Flatten, Embedding
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
import keras.initializers as initializers

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
        in_layer =  Input(shape=in_shape[1])


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

class ConvArchitectureEmb01(Architecture):
    def __init__(self, input_data):
        super(ConvArchitectureEmb01, self).__init__(input_data)
        
    def define_space(self):
        self.space = {
            'conv01_filters' : hp.choice('conv01_filters', [100]),
            'conv01_ksize'   : hp.choice('conv01_ksize',   [7]),
            'conv01_psize'   : hp.choice('conv01_psize',   [0]),

            'conv02_filters' : hp.choice('conv02_filters', [150]),
            'conv02_ksize'   : hp.choice('conv02_ksize',   [21]),
            'conv02_psize'   : hp.choice('conv02_psize',   [12]),

            'dense01' : hp.choice('dense01', [128]),
        }

    def define_architecture(self, in_shape, hp_params):

        print(in_shape)

        p = hp_params
        # Input
        in_layer =  Input(shape=in_shape[1])

        _embedding_dims = 9
        _maxlen = in_shape[0]
        emb = Embedding(4, _embedding_dims, input_length=_maxlen)(in_layer)

        # Filters
        _filters = p['conv01_filters']
        _kernel_size = p['conv01_ksize']
        _pool_size = p['conv01_psize']

        conv = Conv1D(filters=_filters, kernel_size=_kernel_size, activation='relu', name='conv1')(emb)
        if _pool_size > 0:
            conv = MaxPooling1D(pool_size=_pool_size,  name='pool1')(conv)

        _filters = p['conv02_filters']
        _kernel_size = p['conv02_ksize']
        _pool_size = p['conv02_psize']

        conv = Conv1D(filters=_filters, kernel_size=_kernel_size, activation='relu', name='conv2')(conv)
        if _pool_size > 0:
            conv = MaxPooling1D(pool_size=_pool_size,  name='pool2')(conv)

        # Flat
        flat = Flatten()(conv)

        # Dropout
        drop = Dropout(.1)(flat)

        # Fully connect
        dense = Dense(p['dense01'], activation='relu')(drop)

        # Output
        out_layer = Dense(1, activation='sigmoid')(dense)

        return in_layer, out_layer

# =================================================================

class ConvArchitectureHot01(Architecture):
    def __init__(self, input_data):
        super(ConvArchitectureHot01, self).__init__(input_data)
        
    def define_space(self):
        self.space = {
            'conv01_filters' : hp.choice('conv01_filters', [100]),
            'conv01_ksize'   : hp.choice('conv01_ksize',   [7]),
            'conv01_psize'   : hp.choice('conv01_psize',   [0]),

            'conv02_filters' : hp.choice('conv02_filters', [150]),
            'conv02_ksize'   : hp.choice('conv02_ksize',   [21]),
            'conv02_psize'   : hp.choice('conv02_psize',   [12]),

            'dense01' : hp.choice('dense01', [128]),
        }

    def define_architecture(self, in_shape, hp_params):

        print(in_shape)

        p = hp_params
        # Input
        in_layer =  Input(shape=in_shape[1])

        _embedding_dims = 9
        _maxlen = in_shape[0]
        emb = Embedding(4, _embedding_dims, input_length=_maxlen)(in_layer)

        # Filters
        _filters = p['conv01_filters']
        _kernel_size = p['conv01_ksize']
        _pool_size = p['conv01_psize']

        conv = Conv1D(filters=_filters, kernel_size=_kernel_size, activation='relu', name='conv1')(emb)
        if _pool_size > 0:
            conv = MaxPooling1D(pool_size=_pool_size,  name='pool1')(conv)

        _filters = p['conv02_filters']
        _kernel_size = p['conv02_ksize']
        _pool_size = p['conv02_psize']

        conv = Conv1D(filters=_filters, kernel_size=_kernel_size, activation='relu', name='conv2')(conv)
        if _pool_size > 0:
            conv = MaxPooling1D(pool_size=_pool_size,  name='pool2')(conv)

        # Flat
        flat = Flatten()(conv)

        # Dropout
        drop = Dropout(.1)(flat)

        # Fully connect
        dense = Dense(p['dense01'], activation='relu')(drop)

        # Output
        out_layer = Dense(1, activation='sigmoid')(dense)

        return in_layer, out_layer

# =================================================================

class ConvArchitectureHot02(Architecture):
    def __init__(self, input_data):
        super(ConvArchitectureHot02, self).__init__(input_data)
        
    def define_space(self):
        self.space = {
            'conv01_filters' : hp.choice('conv01_filters', [100]),
            'conv01_ksize'   : hp.choice('conv01_ksize',   [7]),
            'conv01_psize'   : hp.choice('conv01_psize',   [0]),

            'conv02_filters' : hp.choice('conv02_filters', [150]),
            'conv02_ksize'   : hp.choice('conv02_ksize',   [21]),
            'conv02_psize'   : hp.choice('conv02_psize',   [12]),

            'dense01' : hp.choice('dense01', [128]),
        }

    def default_space(self):
        space = {
            'conv01_filters' : 100,
            'conv01_ksize'   : 7,
            'conv01_psize'   : 0,
            'conv02_filters' : 150,
            'conv02_ksize'   : 21,
            'conv02_psize'   : 12,
            'dense01' : 128,
        }
        return space

    def define_architecture(self, in_shape, hp_params):

        print('in_shape')
        print(in_shape)
        # Verify
        if hp_params == None:
            p = self.default_space
        else:
            p = hp_params

        # Input
        in_layer =  Input(shape=(4, in_shape[2], 1))

        # Filters
        _filters = p['conv01_filters']
        _kernel_size = p['conv01_ksize']
        _pool_size = p['conv01_psize']

        conv = Conv2D(filters=_filters, kernel_size=(4,_kernel_size), activation='relu', name='conv1', kernel_initializer=initializers.he_normal(seed=123))(in_layer)
        if _pool_size > 0:
            conv = MaxPooling2D(pool_size=(1,_pool_size), strides=(1,_pool_size), name='pool1')(conv)

        _filters = p['conv02_filters']
        _kernel_size = p['conv02_ksize']
        _pool_size = p['conv02_psize']

        conv = Conv2D(filters=_filters, kernel_size=(1,_kernel_size), activation='relu', name='conv2')(conv)
        if _pool_size > 0:
            conv = MaxPooling2D(pool_size=(1,_pool_size), strides=(1,_pool_size),  name='pool2')(conv)

        # Flat
        flat = Flatten()(conv)

        # Fully connect
        dense = Dense(p['dense01'], activation='relu')(flat)

        # Output
        out_layer = Dense(1, activation='sigmoid')(dense)

        return in_layer, out_layer

# =================================================================

class ArchitectureFactory(object):
    def __init__(self, input_data):
        self.input_data = input_data

    def get_architecture(self, architecture_type):
        input_shape = self.input_data.shape
        architecture = None
        if architecture_type == 'MLP':
            architecture = MLPArchitecture(input_shape)
        elif architecture_type == 'Conv_emb_01':
            architecture = ConvArchitectureEmb01(input_shape)
        elif architecture_type == 'Conv_hot_01':
            architecture = ConvArchitectureHot01(input_shape)
        elif architecture_type == 'Conv_hot_02':
            architecture = ConvArchitectureHot02(input_shape)
        else:
            print('ERROR: "{}" is not a valid architecture type.'.format(architecture_type))
        return architecture