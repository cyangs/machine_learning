import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import sklearn.model_selection as ms
from assignment1 import exp_runner

class NeuralNetwork:

    def testRun(self):
        return NotImplementedError

    def keplerData(self, kepler_df):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        # print(X, y)

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        mlp = MLPRegressor(hidden_layer_sizes=(13, 13, 13), max_iter=500)
        mlp.fit(X_train, y_train)

        score = mlp.score(X_test, y_test)
        print("[*] Accuracy: ", score)

        # not work on continous data that we have as in Kepler
        # print(confusion_matrix(y_test, predictions))
        # print(classification_report(y_test, predictions))

    def insuranceData(self, insurance_df):
        y = insurance_df['Destination']
        X = insurance_df
        del X['Destination']  # delete from X we don't need it

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        alphas = [2., 4.]
        hiddens = [(h,) * l for l in [1, 2, 3] for h in [4, 4 // 2, 4 * 2]]
        hiddens += [(32,), (64,), (16,), (100, 100, 100), (60, 60, 60)]

        params = {
            'MLP__solver': ['lbfgs', 'adam'],
            'MLP__alpha': alphas,
            'MLP__hidden_layer_sizes': hiddens,
            'MLP__random_state': [1],
            'MLP__activation': ['identity', 'relu']
        }

        learner = Pipeline([('MLP', MLPRegressor(max_iter=1000))])
        gs = ms.GridSearchCV(
            learner,
            params,
            cv=3, n_jobs=2, scoring=None,
            refit='neg_mean_squared_error',
            return_train_score=True, verbose=10
        )

        gs.fit(X_train, y_train)
        params = gs.best_params_

        model = gs.best_estimator_
        model.set_params(**params)

        print("[*] Learned parameters: {}".format(params))
        model.fit(X_train, y_train)

        return model.score(X_test, y_test)


    def kelperDataBetterModel(self, kepler_df):

        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        # print(X, y)

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        alphas = [2., 4.]
        hiddens = [(h,) * l for l in [1, 2, 3] for h in [4, 4 // 2, 4 * 2]]
        hiddens += [(32,), (64,), (16,), (100, 100, 100), (60, 60, 60)]

        params = {
            'MLP__solver': ['lbfgs', 'adam'],
            'MLP__alpha': alphas,
            'MLP__hidden_layer_sizes': hiddens,
            'MLP__random_state': [1],
            'MLP__activation': ['identity', 'relu']
        }

        learner = Pipeline([('MLP', MLPRegressor(max_iter=1000))])
        gs = ms.GridSearchCV(
            learner,
            params,
            cv=3, n_jobs=2, scoring=None,
            refit='neg_mean_squared_error',
            return_train_score=True, verbose=10
        )

        gs.fit(X_train, y_train)
        params = gs.best_params_

        model = gs.best_estimator_
        model.set_params(**params)

        print("[*] Learned parameters: {}".format(params))
        model.fit(X_train, y_train)

        return model.score(X_test, y_test)

    def get_results(self):
        print("[*] NN - Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] NN - Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def __init__(self):
        print("Neural Network, using Kepler/Insurance data set")
        self.kepler_accuracy = self.kelperDataBetterModel(exp_runner.get_kepler_train_test_data())
        self.insurance_accuracy = self.insuranceData(exp_runner.get_insurance_train_test_data())
        self.get_results()

