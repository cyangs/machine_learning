
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from assignment1 import exp_runner


class SupportVectorMachine:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df):
        # Destination is output variables (that we need to predict)
        y = insurance_df['Destination']
        X = insurance_df
        del X['Destination']  # delete from X we don't need it

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        model = SVR(kernel="rbf",
                    degree=3,
                    gamma="auto_deprecated",
                    coef0=0.0, tol=0.001, C=1.0,
                    epsilon=0.1, shrinking=True,
                    cache_size=200, verbose=False,
                    max_iter=-1)

        model.fit(X_train, y_train)
        return model.score(X_test, y_test)


    def keplerData(self, kepler_df):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        model = SVR(kernel="rbf",
                    degree=3,
                    gamma="auto_deprecated",
                    coef0=0.0, tol=0.001, C=1.0,
                    epsilon=0.1, shrinking=True,
                    cache_size=200, verbose=False,
                    max_iter=-1)

        model.fit(X_train, y_train)

        return model.score(X_test, y_test)

    def get_results(self):
        print("[*] SVM - Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] SVM - Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def __init__(self):
        print("Support Vector Machine, using Kepler/Insurance data set")
        self.kepler_accuracy = self.keplerData(exp_runner.get_kepler_train_test_data())
        self.insurance_accuracy = self.insuranceData(exp_runner.get_insurance_train_test_data())
        self.get_results()