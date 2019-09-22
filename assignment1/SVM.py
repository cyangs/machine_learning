
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import SVC
from assignment1 import exp_runner
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
from sklearn.datasets import make_regression
import timeit

class SupportVectorMachine:

    def insuranceLinear(self, insurance_df):
        y = insurance_df['Claim']
        X = insurance_df
        del X['Claim']  # delete from X we don't need it

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        regr = LinearSVC(random_state=0, tol=1e-5)
        start_time = timeit.default_timer()

        regr.fit(X, y)
        end_time = timeit.default_timer()
        accuracy = regr.score(X_test, y_test)
        elapsed_time = end_time - start_time
        return accuracy, elapsed_time

    def insuranceData(self, insurance_df, kernel):
        # Destination is output variables (that we need to predict)
        logging = {}

        if (kernel is "linear"):
            print("Running Linear Kernal...")
            accuracy, elapsed_time = self.insuranceLinear(insurance_df)
            logging['training_time'] = elapsed_time
            logging['accuracy'] = accuracy
            return logging

        y = insurance_df['Claim']
        X = insurance_df
        del X['Claim']  # delete from X we don't need it

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # model = SVC(kernel=kernel,
        #             degree=3,
        #             gamma="auto_deprecated",
        #             coef0=0.0, tol=0.001, C=1.0,
        #             shrinking=True,
        #             cache_size=200, verbose=False,
        #             max_iter=-1)
        model = SVC(kernel=kernel)

        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        accuracy = model.score(X_test, y_test)

        logging['training_time'] = end_time - start_time
        logging['accuracy'] = accuracy
        return logging

    def keplerLinear(self, kepler_df):
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        regr = LinearSVR(random_state=0, tol=1e-5)
        start_time = timeit.default_timer()

        regr.fit(X, y)
        end_time = timeit.default_timer()
        accuracy = regr.score(X_test, y_test)
        elapsed_time = end_time - start_time
        return accuracy, elapsed_time


    def keplerData(self, kepler_df, kernel):
        logging = {}

        if (kernel is "linear"):
            print("Running Linear Kernal...")
            accuracy, elapsed_time = self.keplerLinear(kepler_df)
            logging['training_time'] = elapsed_time
            logging['accuracy'] = accuracy
            return logging

        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


        ## VALUE OF C is important
        ## Try varying C, Gamma, and Epsilon
        ## TRY SK LEARN LinearSVC

        # model = SVR(kernel=kernel,
        #             degree=3,
        #             gamma="auto_deprecated",
        #             coef0=0.0, tol=0.001, C=1.0,
        #             epsilon=0.1, shrinking=True,
        #             cache_size=200, verbose=False,
        #             max_iter=-1)

        model = SVR(kernel=kernel)

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

    def __init__(self):
        print(f"Support Vector Machine: Kepler/Insurance data set")
        kernels = ['rbf', 'poly', 'sigmoid', 'linear']
        self.kepler_graph_data = pd.DataFrame(columns=['kernel', 'accuracy'], index=kernels)
        self.insurance_data = pd.DataFrame(columns=['kernel', 'accuracy'], index=kernels)

        kernels = ['rbf', 'linear', 'sigmoid']

        for kernel in kernels:
            print(f"Starting {kernel}...")
            kepler_df = exp_runner.get_kepler_train_test_data()
            insurance_df = exp_runner.get_insurance_train_test_data()

            self.kepler_logging = self.keplerData(kepler_df, kernel)
            self.kepler_graph_data.loc[kernel].accuracy = self.kepler_logging.get('accuracy')
            print(f"[*][{kernel}] SVM- Kepler Data Accuracy: {self.kepler_logging.get('accuracy')}")

            self.insurance_logging = self.insuranceData(insurance_df, kernel)
            self.insurance_data.loc[kernel].accuracy = self.insurance_logging.get('accuracy')
            print(f"[*][{kernel}] SVM- Insurance Data Accuracy: {self.insurance_logging.get('accuracy')}")

            # self.kepler_graph_data.loc[kernel].runtime = self.kepler_logging.get('training_time')
            # self.insurance_data.loc[kernel].runtime = self.insurance_logging.get('training_time')
            # print(f"[*][{kernel}] SVM- Kepler Training Runtime: {self.kepler_logging.get('training_time')}")
            # print(f"[*][{kernel}] SVM- Insurance Training Runtime: {self.insurance_logging.get('training_time')}")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

