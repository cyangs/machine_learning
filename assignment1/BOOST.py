import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from assignment1 import exp_runner

class Boosting:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df):
        y = insurance_df['Destination']
        X = insurance_df
        del X['Destination']  # delete from X we don't need it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        classifier = ensemble.AdaBoostRegressor(
            base_estimator=None,
            n_estimators=50,
            learning_rate=1,
            random_state=None)

        boost_model = classifier.fit(X_train, y_train)

        # Model Accuracy, how often is the classifier correct?
        return boost_model.score(X_test, y_test)

    def keplerData(self, kepler_df):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        print(set(y))

        # divide X and y into train and test | train on different data | test on different -> good matrix
        # 70% training and 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        classifier = ensemble.AdaBoostRegressor(
            base_estimator=None,
            n_estimators=50,
            learning_rate=1,
            random_state=None)

        boost_model = classifier.fit(X_train, y_train)

        # Model Accuracy, how often is the classifier correct?
        return boost_model.score(X_test, y_test)

    def get_results(self):
        print("[*] Boost - Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] Boost - Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def __init__(self):
        print("Boost, using Kepler/Insurance data set")
        self.kepler_accuracy = self.keplerData(exp_runner.get_kepler_train_test_data())
        self.insurance_accuracy = self.insuranceData(exp_runner.get_insurance_train_test_data())
        self.get_results()
