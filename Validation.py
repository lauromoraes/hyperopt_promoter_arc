#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
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

class Validation(object):
    def __init__(self, architecture):
        self.architecture = architecture

    def setup_folds(self, X, Y, n_splits=2, seed=173):
        kf = StratifiedShuffleSplit(n_splits=n_splits, random_state=seed)
        kf.get_n_splits(X, Y)
        return kf

    def eval(self, model, data):
        from ml_statistics import BaseStatistics
        x_test, y_test = data
        y_pred = model.predict(x_test, batch_size=32)    
        stats = BaseStatistics(y_test, y_pred)
        return stats
        

    def optimize(self, input_data=None):
        X, Y = input_data
        # X = X.reshape(X.shape[0], X.shape[1], 1)
        in_shape = X.shape
        space = self.architecture.get_space()
        folds = self.setup_folds(X, Y)

        def get_calls(partition):
            from keras import callbacks as C
            calls = list()
            calls.append( C.ModelCheckpoint(save_dir +'/'+'-partition_{}'.format(partition)+'-epoch_{epoch:02d}-weights.h5', save_best_only=True, save_weights_only=True, verbose=1) )
            # calls.append( C.CSVLogger(args.save_dir + '/log.csv') )
            # calls.append( C.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs/{}'.format(actual_partition), batch_size=args.batch_size, histogram_freq=args.debug) )
            calls.append( C.EarlyStopping(monitor='val_loss', patience=10, verbose=0) )
            # calls.append( C.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001, verbose=0) )
            calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.98 ** epoch)) )
        #    calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.)) )
            return calls



        def f(hp_params):
            nsplits = 5
            cnt1 = 0
            scores, score = [], None
            seed = 123

            print('Testing parameters')
            print(hp_params)
            if len(trials.trials)>1:
                for x in trials.trials[:-1]:
                    space_point_index = dict([(key,value[0]) for key,value in x['misc']['vals'].items() if len(value)>0])
                    peval = space_eval(space,space_point_index)
                    if hp_params == peval:
                        print('>>> Repeated Evaluation')
                        loss = x['result']['loss']
                        return {'loss':loss, 'status':STATUS_FAIL}


            # Folds for train - test (Evaluate Model)
            folds1 = StratifiedShuffleSplit(n_splits=nsplits, random_state=seed)
            for train_index, test_index in folds1.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                cnt1 += 1

                # Folds for train - validation (Guide training phase)
                folds2 = StratifiedShuffleSplit(n_splits=1, random_state=seed, test_size=0.01)
                for t_index, v_index in folds2.split(X_train, Y_train):
                    x_train, x_val = X_train[t_index], X_train[v_index]
                    y_train, y_val = Y_train[t_index], Y_train[v_index]
                    val_data=(x_val, y_val)

                    in_layer, out_layer = self.architecture.define_architecture(in_shape, hp_params)
                    model = Model([in_layer], [out_layer])
                    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy')
                    # if cnt1 == 1:
                    #     model.summary()

                    # Update partition couter
                    print('Partition', cnt1)

                    # Get Updated Callbacks
                    calls = get_calls(cnt1)

                    # Train model
                    model.fit(X_train, Y_train, epochs=300, batch_size=32, verbose=0,callbacks=calls, validation_data=val_data)

                    stats = self.eval(model, (X_test,Y_test))
                    score = -stats.Mcc
                    scores.append(score)
                    print('score', score)

                    K.clear_session()
                    del model

#            for t_index, v_index in folds.split(X, Y):
#                X_train, X_val = X[t_index], X[v_index]
#                Y_train, Y_val = Y[t_index], Y[v_index]      
#                val_data=(X_val, Y_val)
#
#                calls = get_calls()
#
#                in_layer, out_layer = self.architecture.define_architecture(in_shape, hp_params)
#                model = Model([in_layer], [out_layer])
#                model.compile(optimizer='adam', loss='binary_crossentropy')
#                print(model.summary)
#                
#                model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1,callbacks=calls, validation_data=val_data)
#
#                stats = self.eval(model, (X_train,Y_train))
#                score = stats.Mcc
#                scores.append(score)
#
#                K.clear_session()
#                del model

            final_score = np.mean(scores)
            print('scores', scores)
            print('final_score', final_score)
            print("")
            return {'loss':final_score, 'status':STATUS_OK}

        trials = Trials()

        best = fmin(f, space, algo=tpe.suggest, trials=trials, max_evals=100)
        best_params = space_eval(space, best)

        return best, best_params
