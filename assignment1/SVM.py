
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import SVC
from assignment1 import exp_runner
import timeit

class SupportVectorMachine:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df, kernel):
        # Destination is output variables (that we need to predict)
        y = insurance_df['Gender']
        X = insurance_df
        del X['Gender']  # delete from X we don't need it

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        model = SVC(kernel=kernel,
                    degree=3,
                    gamma="auto_deprecated",
                    coef0=0.0, tol=0.001, C=1.0,
                    epsilon=0.1, shrinking=True,
                    cache_size=200, verbose=False,
                    max_iter=-1)

        logging = {}
        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = accuracy
        return logging


    def keplerData(self, kepler_df, kernel):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        model = SVR(kernel=kernel,
                    degree=3,
                    gamma="auto_deprecated",
                    coef0=0.0, tol=0.001, C=1.0,
                    epsilon=0.1, shrinking=True,
                    cache_size=200, verbose=False,
                    max_iter=-1)

        logging = {}

        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        accuracy = model.score(X_test, y_test)
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = accuracy
        return logging


    def get_kepler_results(self):
        return self.kepler_graph_data

    def get_insurance_results(self):
        return self.insurance_data

    def __init__(self, variance = 'kernel'):
        print(f"Support Vector Machine, using {variance} and Kepler/Insurance data set")
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        self.kepler_graph_data = pd.DataFrame(columns=['kernel', 'accuracy', 'runtime'], index=kernels)
        self.insurance_data = pd.DataFrame(columns=['kernel', 'accuracy', 'runtime'], index=kernels)

        for kernel in kernels:
            kepler_df = exp_runner.get_kepler_train_test_data()
            insurance_df = exp_runner.get_insurance_train_test_data()

            self.kepler_logging = self.keplerData(kepler_df, kernel)
            self.kepler_graph_data.loc[kernel].accuracy = self.kepler_logging.get('accuracy')
            self.kepler_graph_data.loc[kernel].runtime = self.kepler_logging.get('training_time')
            print(f"[*][{kernel}] SVM- Kepler Data Accuracy: {self.kepler_logging.get('accuracy')}")
            print(f"[*][{kernel}] SVM- Kepler Training Runtime: {self.kepler_logging.get('training_time')}")

            self.insurance_logging = self.insuranceData(insurance_df, kernel)
            self.insurance_data.loc[kernel].accuracy = self.insurance_logging.get('accuracy')
            self.insurance_data.loc[kernel].runtime = self.insurance_logging.get('training_time')
            print(f"[*][{kernel}] SVM- Insurance Data Accuracy: {self.insurance_logging.get('accuracy')}")
            print(f"[*][{kernel}] SVM- Insurance Training Runtime: {self.insurance_logging.get('training_time')}")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

