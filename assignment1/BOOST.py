from sklearn import ensemble
from sklearn.model_selection import train_test_split
from assignment1 import exp_runner
from sklearn.model_selection import cross_val_score
from sklearn import tree
import numpy as np
import pandas as pd
import timeit

class Boosting:

    def insuranceData(self, insurance_df, n_estimators = 50, learning_rate = 1, max_depth = 1, DT = False):
        y = insurance_df['Claim']
        X = insurance_df
        del X['Claim']  # delete from X we don't need it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        if DT:
            classifier = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth),
                                        n_estimators=50)
        else:
            classifier = ensemble.AdaBoostClassifier(
                base_estimator=None,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=None)

        logging = {}
        start_time = timeit.default_timer()
        boost_model = classifier.fit(X_train, y_train)
        end_time = timeit.default_timer()

        # Model Accuracy, how often is the classifier correct?
        accuracy = boost_model.score(X_test, y_test)
        cross_val = cross_val_score(boost_model, X, y, cv=3)
        avg_score = np.mean(cross_val)

        # from sklearn.metrics import confusion_matrix
        # y_pred = classifier.fit(X_train, y_train).predict(X_test)
        # cm = confusion_matrix(y_test, y_pred)

        logging['training_time'] = end_time - start_time
        logging['accuracy'] = accuracy
        logging['cross_val'] = avg_score
        return logging

    def keplerData(self, kepler_df, n_estimators = 50, learning_rate = 1, max_depth = 1, DT = False):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        print(set(y))

        # divide X and y into train and test | train on different data | test on different -> good matrix
        # 70% training and 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Learning rate shrinks the contribution of each classifier by learning Rate. There is a trade off
        # between learning rate and n-estimators.
        if DT:
            classifier = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth),
                                        n_estimators=50)
        else:
            classifier = ensemble.AdaBoostRegressor(
                base_estimator=None,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=None)

        logging = {}

        start_time = timeit.default_timer()
        boost_model = classifier.fit(X_train, y_train)
        cross_val = cross_val_score(boost_model, X, y, cv=3)
        avg_score = np.mean(cross_val)
        accuracy = boost_model.score(X_test, y_test)

        end_time = timeit.default_timer()
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = accuracy
        logging['cross_val'] = avg_score
        return logging

    def get_results(self):
        print("[*] Boost - Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] Boost - Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def get_kepler_results(self):
        return self.kepler_graph_data

    def get_insurance_results(self):
        return self.insurance_data

    def __init__(self,  estimators = 0, variance = "n_estimators"):
        print("Boost, using Kepler/Insurance data set")
        print(f"Estimators: {estimators}. Using {variance} as the treatment.")
        self.variance = variance
        self.kepler_graph_data = pd.DataFrame(columns=['estimators', 'cross_val', 'accuracy', 'runtime'], index=range(estimators))
        self.insurance_data = pd.DataFrame(columns=['estimators', 'cross_val', 'accuracy', 'runtime'], index=range(estimators))

        for i in range(estimators):
            if i == 0:
                continue

            kepler_df = exp_runner.get_kepler_train_test_data()
            insurance_df = exp_runner.get_insurance_train_test_data()

            if self.variance == 'n_estimators':
                self.kepler_logging = self.keplerData(kepler_df, i)
                self.insurance_logging = self.insuranceData(insurance_df, i)
            elif self.variance == 'learning_rate':
                self.kepler_logging = self.keplerData(kepler_df, 50, i)
                self.insurance_logging = self.insuranceData(insurance_df, 50, i)
            elif self.variance == 'maximum_depth':
                self.kepler_logging = self.keplerData(kepler_df, 50, i, True)
                self.insurance_logging = self.insuranceData(insurance_df, 50, i, True)
            else:
                print("Not a valid learning parameter")
                pass

            self.kepler_graph_data.loc[i].estimators = i
            self.kepler_graph_data.loc[i].accuracy = self.kepler_logging.get('accuracy')
            self.kepler_graph_data.loc[i].runtime = self.kepler_logging.get('training_time')
            self.kepler_graph_data.loc[i].cross_val = self.kepler_logging.get('cross_val')
            print(f"[*][{i}] BOOST- Kepler Data Accuracy: {self.kepler_logging.get('accuracy')}")
            print(f"[*][{i}] BOOST- Kepler Training Runtime: {self.kepler_logging.get('training_time')}")
            print(f"[*][{i}] BOOST- Kepler Cross Validation: {self.kepler_logging.get('cross_val')}")

            self.insurance_data.loc[i].estimators = i
            self.insurance_data.loc[i].accuracy = self.insurance_logging.get('accuracy')
            self.insurance_data.loc[i].runtime = self.insurance_logging.get('training_time')
            self.insurance_data.loc[i].cross_val = self.insurance_logging.get('cross_val')
            print(f"[*][{i}] BOOST- Insurance Data Accuracy: {self.insurance_logging.get('accuracy')}")
            print(f"[*][{i}] BOOST- Insurance Training Runtime: {self.insurance_logging.get('training_time')}")
            print(f"[*][{i}] BOOST- Insurance Cross Validation: {self.insurance_logging.get('cross_val')}")

