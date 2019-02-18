#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from hyperopt.mongoexp import MongoTrials

from keras import optimizers
from keras.models import Model
from keras import backend as K

from sklearn.model_selection import StratifiedShuffleSplit

save_dir='./result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def load_dataset2d(organism):
    from ml_data import SequenceNucHotvector

    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)

    samples = SequenceNucHotvector(npath, ppath)

    X, y = samples.getX(), samples.getY()
    y = y.astype('int32')
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    return X, y

class Results(object):
    def __init__(self):
        self.metrics, self.results = self.init_results()

    def init_results(self):
        metrics = ('partition', 'mcc', 'f1', 'sn', 'sp', 'acc', 'prec', 'tp', 'fp', 'tn', 'fn')
        results = { k : [] for k in metrics }
        return metrics, results

    def allocate_stats(self, stats, partition):
        results = self.results
        results['partition'].append(partition)
        results['mcc'].append(stats.Mcc)
        results['f1'].append(stats.F1)
        results['sn'].append(stats.Sn)
        results['sp'].append(stats.Sp)
        results['acc'].append(stats.Acc)
        results['prec'].append(stats.Prec)
        results['tp'].append(stats.tp)
        results['fp'].append(stats.fp)
        results['tn'].append(stats.tn)
        results['fn'].append(stats.fn)
        return results

    def allocate_summarization(self):
        # Foreach calculated metric
        for k, v in self.results.items():
            # Calculate mean and std of partitions results
            M = np.mean(self.results[k])
            D = np.std(self.results[k])
            # Allocate new values on respective column
            self.results[k].append( M )
            self.results[k].append( D )

class Validation(object):
    def __init__(self, architecture, organism):
        self.organism = organism
        self.results = Results()
        self.architecture = architecture
        self.setup_parameters()

    def setup_parameters(self, lr=0.001, lr_decay=.9, batch_size=32, epochs=100, debug=1):
        self.debug = debug
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.stop_patience = 10

    def eval(self, model, data):
        from ml_statistics import BaseStatistics
        x_test, y_test = data
        y_pred = model.predict(x_test, batch_size=32)    
        stats = BaseStatistics(y_test, y_pred)
        return stats
        
    def get_calls(self, partition, seed):
        from keras import callbacks as C
        
        fname = save_dir+'/'+'org_{}-partition_{}-seed_{}'.format(self.organism, partition, seed)+'-epoch_{epoch:02d}-weights.h5'
        
        calls = list()
        calls.append( C.ModelCheckpoint(fname, save_best_only=True, save_weights_only=True, verbose=1) )
        calls.append( C.CSVLogger(save_dir+'/log.csv') )
        calls.append( C.TensorBoard(log_dir=save_dir+'/tensorboard-logs/{}'.format(partition), batch_size=self.batch_size, histogram_freq=self.debug) )
        calls.append( C.EarlyStopping(monitor='val_loss', patience=self.stop_patience, verbose=0) )
        # calls.append( C.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001, verbose=0) )
        calls.append( C.LearningRateScheduler(schedule=lambda epoch: self.lr * (self.lr_decay ** epoch)) )
    #    calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.)) )
        return calls

    def get_best_model(self, partition, x_test, y_test):
        
        # Define prefix and sufix filenames      
        file_prefix = 'org_{}-partition_{}'.format(self.organism, partition)
        file_sufix = '-weights.h5'
        
        # Match all weight files in directory
        model_weights = [ x for x in os.listdir(save_dir+'/') if x.startswith(file_prefix) and x.endswith(file_sufix) ]
        print 'Testing weigths', model_weights

        # Setup local variables
        best_mcc = -10000.0
        selected_weight = None
        selected_stats = None

        # Foreach finded file
        for i in range(len(model_weights)):

            # Select a weight file
            weight_file = model_weights[i]

            # Get input and output
            in_layer, out_layer = self.architecture.define_architecture(self.in_shape, None)

            # Setup model
            model = Model([in_layer], [out_layer])

            # Load weight
            model.load_weights(save_dir + '/' + weight_file)

            stats = self.eval(model=model, data=(x_test, y_test))

            print('Testing file {}'.format(weight_file))
            print('MCC = {}'.format(stats.Mcc))
            
            # Get current best weigth
            if best_mcc < stats.Mcc:
                best_mcc = stats.Mcc
                selected_weight = weight_file
                selected_stats = stats
                print('Selected BEST')
                print(stats)

        print('BEST MODEL for partition {}'.format(partition))
        print(selected_weight)

        # Persist best weights
        in_layer, out_layer = self.architecture.define_architecture(self.in_shape, None)
        model = Model([in_layer], [out_layer])
        model.load_weights(save_dir + '/' + selected_weight)
        model.save_weights(save_dir + '/org_{}-partition_{}-batch_{}-best_weights.h5'.format(self.organism, partition, self.batch_size))
        
        K.clear_session()
        del model
        
        # Delete others weights
        for i in range(len(model_weights)):
            weight_file = model_weights[i]
            print('Deleting weight: {}'.format(weight_file))
            path = save_dir + '/' + weight_file
            try:
                os.remove(path)
            except:
                pass

        return (selected_stats, selected_weight)



    def crossval_model(self, input_data=None, nsplits=5, seed=61):
        X, Y = input_data

        # X = X.reshape(X.shape[0], X.shape[1], 1)
        self.in_shape = X.shape

        # Setup partition counter
        partition_counter = 0

        # Folds for train - test (Evaluate Model)
        folds1 = StratifiedShuffleSplit(n_splits=nsplits, random_state=seed)
        for train_index, test_index in folds1.split(X, Y):

            # Define indexes for training-val set and test set
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            # Update partition counter
            partition_counter += 1

            # ==== Transfer Learning ====
            # transfer = load_dataset2d('Bacillus')
            # Xt, Yt = transfer
            # # Get input and output
            # in_layer, out_layer = self.architecture.define_architecture(self.in_shape, None)
            # # Setup model
            # model = Model([in_layer], [out_layer])
            # # Compile model
            # model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy')
            # # Get Updated Callbacks
            # calls = self.get_calls(partition_counter, seed)
            # # Train model
            # model.fit(Xt, Yt, epochs=self.epochs, batch_size=self.batch_size, verbose=0,callbacks=calls)


            # Train on different initialization seeds
            seeds = [23, 29, 31]
            for seed in seeds:
                print('Partition {} - Training on SEED {}'.format(partition_counter, seed))

                # Folds for train - validation (Guide training phase)
                folds2 = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
                for t_index, v_index in folds2.split(X_train, Y_train):

                    # Define indexes for training set and validation set
                    x_train, x_val = X_train[t_index], X_train[v_index]
                    y_train, y_val = Y_train[t_index], Y_train[v_index]
                    val_data=(x_val, y_val)

                    # Get input and output
                    in_layer, out_layer = self.architecture.define_architecture(self.in_shape, None)

                    # Setup model
                    model = Model([in_layer], [out_layer])

                    # Compile model
                    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy')

                    # Print network
                    if partition_counter == 1:
                        model.summary()

                    # Update partition couter
                    print('Partition', partition_counter)

                    # Get Updated Callbacks
                    calls = self.get_calls(partition_counter, seed)

                    # Train model
                    model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0,callbacks=calls, validation_data=val_data)

                    # Evaluate model on test set
                    stats = self.eval(model, (X_test,Y_test))

                    # Allocate metrics
                    score = stats.Mcc
                    print('score', score)

                    # Prepare to receive new model
                    K.clear_session()
                    del model

            # Select best model
            (stats, weight_file) = self.get_best_model(partition_counter, X_test, Y_test)
            print('Selected BEST: {} ({})'.format(weight_file, stats.Mcc))

            self.results.allocate_stats(stats, partition_counter)

        self.results.allocate_summarization()
        # Write results of partitions to CSV
        df = pd.DataFrame(self.results.results, columns=self.results.metrics)
        print(df)
        df.to_csv('results_org-{}_batch-{}'.format(self.organism, self.batch_size ))
