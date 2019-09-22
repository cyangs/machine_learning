import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import sklearn.model_selection as ms
from assignment1 import exp_runner
import timeit

class NeuralNetwork:

    def insuranceData(self, insurance_df):
        logging = {}
        start_time = timeit.default_timer()
        y = insurance_df['Claim']
        X = insurance_df
        del X['Claim']  # delete from X we don't need it

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

        end_time = timeit.default_timer()
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = model.score(X_test, y_test)
        return logging


    def kelperDataBetterModel(self, kepler_df):
        logging = {}
        start_time = timeit.default_timer()

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

        start_time = timeit.default_timer()
        gs.fit(X_train, y_train)
        params = gs.best_params_

        model = gs.best_estimator_
        model.set_params(**params)

        print("[*] Learned parameters: {}".format(params))

        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = model.score(X_test, y_test)
        return logging

    def get_kepler_results(self):
        return self.kepler_logging

    def get_insurance_results(self):
        return

    def get_results(self):
        print("[*] NN - Kepler Data Accuracy: {}".format(self.kepler_logging.get('accuracy')))
        print("[*] NN - Kepler Training Time: {}".format(self.kepler_logging.get('training_time')))

        print("[*] NN - Insurance Data Accuracy: {}".format(self.insurance_logging.get('accuracy')))
        print("[*] NN - Insurance Training Time: {}".format(self.insurance_logging.get('training_time')))

    def __init__(self):
        print("Neural Network, using Kepler/Insurance data set")
        # self.kepler_graph_data = pd.DataFrame(columns=['max_depth', 'accuracy', 'runtime'], index=range(max_depth))
        # self.insurance_data = pd.DataFrame(columns=['max_depth', 'accuracy', 'runtime'], index=range(max_depth))
        kepler_df = exp_runner.get_kepler_train_test_data()
        insurance_df = exp_runner.get_insurance_train_test_data()

        self.kepler_logging = self.kelperDataBetterModel(kepler_df)
        self.insurance_logging = self.insuranceData(insurance_df)
        self.get_results()

