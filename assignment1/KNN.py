from sklearn import neighbors
from sklearn.model_selection import train_test_split
from assignment1 import exp_runner
import pandas as pd
import matplotlib.pyplot as plt
import timeit


class KNearestNeighbor:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df, n_neighbors = 5, leaf_size = 30, algorithm = 'auto'):
        # Destination is output variables (that we need to predict)
        y = insurance_df['Gender']
        X = insurance_df
        del X['Gender']  # delete from X we don't need it

        # X = X._get_numeric_data
        # print(X, y)

        model = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='uniform',
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=1
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        training_time = end_time - start_time

        accuracy = model.score(X_test, y_test)
        return accuracy, training_time


    def keplerData(self, kepler_df, n_neighbors = 5, leaf_size = 30, algorithm = 'auto'):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        # if self.verbose:
        #     print(X, y)

        print(f"Running with Nneighbors: {n_neighbors}, Leaf Size: {leaf_size}, Algo: {algorithm}")

        model = neighbors.KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights='uniform',
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=1
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        training_time = end_time - start_time

        # accuracy
        accuracy = model.score(X_test, y_test)
        return accuracy, training_time

    def get_results(self):
        print("[*] KNN- Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] KNN- Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def get_kepler_results(self):
        return self.kepler_graph_data

    def get_insurance_results(self):
        return self.insurance_data

    def __init__(self, runs = 0):
        print("k-nearest neighbor, using Kepler/Insurance data set")
        print(f"Runs: {runs}")
        self.kepler_graph_data = pd.DataFrame(columns=['runs', 'accuracy', 'runtime'], index=range(runs))
        self.insurance_data = pd.DataFrame(columns=['runs', 'accuracy', 'runtime'], index=range(runs))

        for i in range(runs):
            if i == 0:
                continue
            kepler_df = exp_runner.get_kepler_train_test_data()
            self.kepler_accuracy, self.kepler_runtime = self.keplerData(kepler_df, i)
            self.kepler_graph_data.loc[i].runs = i
            self.kepler_graph_data.loc[i].accuracy = self.kepler_accuracy
            print(f"[*][{i}] KNN- Kepler Data Accuracy: {self.kepler_accuracy}")
            print(f"[*][{i}] KNN- Kepler Training Runtime: {self.kepler_runtime}")

            insurance_df = exp_runner.get_insurance_train_test_data()
            self.insurance_accuracy, self.insurance_runtime = self.insuranceData(insurance_df, i)
            self.insurance_data.loc[i].runs = i
            self.insurance_data.loc[i].accuracy = self.insurance_accuracy
            print(f"[*][{i}] KNN- Insurance Data Accuracy: {self.insurance_accuracy}")
            print(f"[*][{i}] KNN- Insurance Training Runtime: {self.insurance_runtime}")


