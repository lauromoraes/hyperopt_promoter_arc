#!/usr/bin/python
 # -*- coding: utf-8 -*-
import numpy as np
from hyperopt import hp

from keras import layers, models
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Embedding, Reshape
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, UpSampling2D, UpSampling1D, AveragePooling1D, AveragePooling2D
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
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

        # print(in_shape)

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
        
    def define_space(self, num_branches, depth=1):
        levels = []

        for level in range(depth):
            s = {'depth':level}
            s['conv_{}_num_filters'.format(level)] = hp.choice('conv_filters', [0, 128, 256, 512, 1024])
            # TODO
            s['conv_{}_kernel_shape'.format(level)] = hp.choice('conv_shapes', [(2,2), ()])
            s['conv_{}_pool_shape'.format(level)] = hp.choice('conv_filters', [0, 128, 256, 512, 1024])

            levels.append(s)

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

    def set_branches(self, inputs_shapes, params):
        p = params

        # Define multiple braches of filters
        banches_input, banches_output, cnt = [], [], 0
        for shape in inputs_shapes:
            cnt+=1
            _input = Input(shape=shape, name='input_{}'.format(cnt))
            banches_input.append(_input)


            if p['conv01_filters'] > 0:
                _num_filters = p['conv{}_num_filters'.format(cnt)]
                _kernel_shape = p['conv{}_kernel_shape'.format(cnt)]
                _pool_shape = p['conv_{}_psize'.format(cnt)]
                _layer_name = 'conv_{}'
                _conv1 = Conv2D(
                    filters=_num_filters, 
                    kernel_size=_kernel_shape,
                    activation='relu', 
                    name='conv1'
                )(in_layer)

        _filters = p['conv01_filters']
        _kernel_size = p['conv01_ksize']
        _pool_size = p['conv01_psize']


    def define_architecture(self, inputs_shapes, hp_params):

        # print('in_shape')
        # print(in_shape)
        # Verify
        if hp_params == None:
            p = self.default_space()
        else:
            p = hp_params

        # Define multiple inputs
        inputs = []
        cnt = 0
        for shape in inputs_shapes:
            inputs.append(Input(shape=shape, name='input_{}'.format(cnt)))
            cnt+=1

        # Filters
        _filters = p['conv01_filters']
        _kernel_size = p['conv01_ksize']
        _pool_size = p['conv01_psize']

        conv = Conv2D(filters=_filters, kernel_size=(4,_kernel_size), activation='relu', name='conv1')(in_layer)
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
        dense = Dense(p['dense01'], activation='relu', name='fully01')(flat)

        # Output
        out_layer = Dense(1, activation='sigmoid', name='output_node')(dense)

        return in_layer, out_layer

# =================================================================

class CapsnetArchitectureHot01(Architecture):
    def __init__(self, input_data):
        super(CapsnetArchitectureHot01, self).__init__(input_data)
        
    def define_space(self):
        self.space = {
            'routings' : hp.choice('routings', [1, 2, 3]),

            'conv01_filters' : hp.choice('conv01_filters', [128, 256, 512]),
            'conv01_ksize'   : hp.choice('conv01_ksize',   [3, 5, 7, 9, 11, 13, 21]),
            'conv02_ksize'   : hp.choice('conv02_ksize',   [3, 5, 7, 9, 11, 13, 21]),

            'dim_capsule1' : hp.choice('dim_capsule1', [2, 4, 8]), # 8
            'dim_capsule2' : hp.choice('dim_capsule2', [4, 8, 16]), # 16
            'n_channels' : hp.choice('n_channels', [8, 16, 32]), # 32
        }

    def default_space(self):
        space = {
            'routings' : 3,
        }
        return space

    def define_architecture(self, in_shape, hp_params):

        # print('in_shape')
        # print(in_shape)
        # Verify
        n_class = 1 # Binary classification
        print(hp_params)
        if hp_params == None:
            p = self.default_space()
        else:
            p = hp_params

        routings = p['routings']
        conv01_filters = p['conv01_filters']
        conv01_ksize = p['conv01_ksize']
        conv02_ksize = p['conv02_ksize']
        dim_capsule1 = p['dim_capsule1']
        dim_capsule2 = p['dim_capsule2']
        n_channels = p['n_channels']

        in_shape = in_shape[1:]

        maxlen = in_shape[0]

        # Input
        in_layer =  Input(shape=(in_shape[0], in_shape[1],1), name='input_layer')

        conv1 = Conv2D(filters=conv01_filters, kernel_size=(4,conv01_ksize), strides=1, padding='valid', activation='relu', name='conv1')(in_layer)
        conv1 = AveragePooling2D((1,3))(conv1)
        primarycaps = PrimaryCap(conv1, dim_capsule=dim_capsule1, n_channels=n_channels, kernel_size=(1,conv02_ksize), strides=2, padding='valid')
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule2, routings=routings, name='digitcaps')(primarycaps)
        out_caps = Length(name='capsnet')(digitcaps)

        # Decoder network.
        y = Input(shape=(n_class, ), name='input_decoder')
        masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        # masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        x_recon = layers.Dense(512, activation='relu', name='decode_dense_01')(masked_by_y)
        x_recon = layers.Dense(1024, activation='relu', name='decode_dense_02')(x_recon)
        # # x_recon = layers.Dropout(.2, name='decode_drop_01')(x_recon)
        x_recon = layers.Dense(np.prod(in_shape), activation='sigmoid', name='decode_dense_03')(x_recon)
        x_recon = layers.Reshape(target_shape=in_shape, name='out_recon')(x_recon)

        # Shared Decoder model in training and prediction
        # decoder = Sequential(name='decoder')
        # decoder.add(Dense(512, activation='relu', input_dim=16*n_class))
        # decoder.add(Dense(1024, activation='relu'))
        # x_recon = layers.Dense(maxlen, activation='sigmoid')(x_recon)
        # decoder.add(Dense(np.prod((in_shape[1], in_shape[2],1)), activation='sigmoid'))
        # decoder.add(Reshape(target_shape=(in_shape[1], in_shape[2],1), name='out_recon'))

        # train_model = Model([in_layer, y], [out_caps, decoder(masked_by_y)])
        train_model = Model([in_layer, y], [out_caps, x_recon])


        return train_model

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
        elif architecture_type == 'Capsnet_hot_01':
            architecture = CapsnetArchitectureHot01(input_shape)
        else:
            print('ERROR: "{}" is not a valid architecture type.'.format(architecture_type))
        return architecture