from sklearn import ensemble
from sklearn.model_selection import train_test_split
from assignment1 import exp_runner
import pandas as pd
import timeit

class Boosting:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df, n_estimators = 50, learning_rate = 1):
        y = insurance_df['Gender']
        X = insurance_df
        del X['Gender']  # delete from X we don't need it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = boost_model.score(X_test, y_test)
        return logging

    def keplerData(self, kepler_df, n_estimators = 50, learning_rate = 1):
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

        classifier = ensemble.AdaBoostRegressor(
            base_estimator=None,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=None)

        logging = {}

        start_time = timeit.default_timer()
        boost_model = classifier.fit(X_train, y_train)
        end_time = timeit.default_timer()
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = boost_model.score(X_test, y_test)

        # Model Accuracy, how often is the classifier correct?
        accuracy = boost_model.score(X_test, y_test)
        return logging

    def get_results(self):
        print("[*] Boost - Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] Boost - Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def get_kepler_results(self):
        return self.kepler_graph_data

    def get_insurance_results(self):
        return self.insurance_data

    def __init__(self,  runs = 0, variance = "n_estimators"):
        print("Boost, using Kepler/Insurance data set")
        print(f"Runs: {runs}. Using {variance} as the treatment.")
        self.variance = variance
        self.kepler_graph_data = pd.DataFrame(columns=['runs', 'accuracy', 'runtime'], index=range(runs))
        self.insurance_data = pd.DataFrame(columns=['runs', 'accuracy', 'runtime'], index=range(runs))

        for i in range(runs):
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
            else:
                print("Not a valid learning parameter")
                pass

            self.kepler_graph_data.loc[i].runs = i
            self.kepler_graph_data.loc[i].accuracy = self.kepler_logging.get('accuracy')
            self.kepler_graph_data.loc[i].runtime = self.kepler_logging.get('training_time')
            print(f"[*][{i}] BOOST- Kepler Data Accuracy: {self.kepler_logging.get('accuracy')}")

            self.insurance_data.loc[i].runs = i
            self.insurance_data.loc[i].accuracy = self.insurance_logging.get('accuracy')
            self.insurance_data.loc[i].runtime = self.insurance_logging.get('training_time')
            print(f"[*][{i}] BOOST- Insurance Data Accuracy: {self.insurance_logging.get('accuracy')}")

