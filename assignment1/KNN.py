from sklearn import neighbors
from sklearn.model_selection import train_test_split
from assignment1 import exp_runner
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from sklearn.model_selection import cross_val_score
import numpy as np

class KNearestNeighbor:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df, n_neighbors = 5, leaf_size = 30, algorithm = 'auto'):
        # Destination is output variables (that we need to predict)
        y = insurance_df['Claim']
        X = insurance_df
        del X['Claim']  # delete from X we don't need it

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

        cross_val = cross_val_score(model, X, y, cv=3)
        avg_score = np.mean(cross_val)
        accuracy = model.score(X_test, y_test)
        return accuracy, training_time, avg_score


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
        cross_val = cross_val_score(model, X, y, cv=3)
        avg_score = np.mean(cross_val)
        accuracy = model.score(X_test, y_test)
        return accuracy, training_time, avg_score

    def get_results(self):
        print("[*] KNN- Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] KNN- Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def get_kepler_results(self):
        return self.kepler_graph_data

    def get_insurance_results(self):
        return self.insurance_data

    def __init__(self, neighbors):
        print("k-nearest neighbor, using Kepler/Insurance data set")
        print(f"Runs: {neighbors}")
        self.kepler_graph_data = pd.DataFrame(columns=['neighbors', 'accuracy', 'runtime'], index=range(neighbors))
        self.insurance_data = pd.DataFrame(columns=['neighbors', 'accuracy', 'runtime'], index=range(neighbors))

        for i in range(neighbors):
            if i == 0:
                continue
            kepler_df = exp_runner.get_kepler_train_test_data()
            insurance_df = exp_runner.get_insurance_train_test_data()

            self.kepler_accuracy, self.kepler_runtime, self.kepler_cross_val = self.keplerData(kepler_df, i)
            self.kepler_graph_data.loc[i].neighbors = i
            self.kepler_graph_data.loc[i].accuracy = self.kepler_accuracy
            self.kepler_graph_data.loc[i].runtime = self.kepler_runtime
            self.kepler_graph_data.loc[i].cross_val = self.kepler_cross_val
            print(f"[*][{i}] KNN- :: Kepler Data Accuracy: {self.kepler_accuracy}")
            print(f"[*][{i}] KNN- :: Kepler Cross Validation: {self.kepler_cross_val}")
            print(f"[*][{i}] KNN- :: Kepler Training Runtime: {self.kepler_runtime}")

            self.insurance_accuracy, self.insurance_runtime, self.kepler_cross_val = self.insuranceData(insurance_df, i)
            self.insurance_data.loc[i].neighbors = i
            self.insurance_data.loc[i].accuracy = self.insurance_accuracy
            self.insurance_data.loc[i].runtime = self.insurance_runtime
            self.insurance_data.loc[i].cross_val = self.insurance_cross_val

            print(f"[*][{i}] KNN- Insurance Data Accuracy: {self.insurance_accuracy}")
            print(f"[*][{i}] KNN- Insurance Cross Validation: {self.insurance_cross_val}")
            print(f"[*][{i}] KNN- Insurance Training Runtime: {self.insurance_runtime}")







