import pandas as pd
from assignment1 import exp_runner
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import timeit

class DecisionTree:

    def testRunWithIrisData(self):
        # Load data and store it into pandas DataFrame objects
        iris = load_iris()
        X = pd.DataFrame(iris.data[:, :], columns=iris.feature_names[:])
        y = pd.DataFrame(iris.target, columns=["Species"])

        # Defining and fitting a DecisionTreeClassifier instance
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)

        # Creates dot file named tree.dot
        export_graphviz(
            tree,
            out_file="../output/IrisOutput_DT.dot",
            feature_names=list(X.columns),
            class_names=iris.target_names,
            filled=True,
            rounded=True)

        sample_one_pred = int(tree.predict([[5, 5, 1, 3]]))
        sample_two_pred = int(tree.predict([[5, 5, 2.6, 1.5]]))
        print(f"The first sample most likely belongs a {iris.target_names[sample_one_pred]} flower.")
        print(f"The second sample most likely belongs a {iris.target_names[sample_two_pred]} flower.")

    def insuranceData(self, insurance_df, max_depth):
        # Destination is output variables (that we need to predict)

        # y = insurance_df['Age']
        # X = insurance_df
        # del X['Age']  # delete from X we don't need it

        y = insurance_df['Gender']
        X = insurance_df
        del X['Gender']  # delete from X we don't need it

        # print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # test_size = 0.25: Means 25% data is in test and 75% in train (you can try to vary it)

        # regressor because output variable is not 0 or 1 but continous
        model = tree.DecisionTreeClassifier(criterion="entropy",
                                            splitter="best",
                                            max_depth=max_depth,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            max_features=None,
                                            max_leaf_nodes=None,
                                            random_state=100)

        # model = GradientBoostingClassifier(
        #     loss="deviance", learning_rate=0.01, n_estimators=1000, subsample=0.3,
        #     criterion="friedman_mse", min_samples_split=5, min_samples_leaf=1,
        #     min_weight_fraction_leaf=0.0, max_depth=8, min_impurity_decrease=0.0,
        #     min_impurity_split=None, init=None, random_state=None, max_features=None,
        #     verbose=1, n_iter_no_change=None, tol=0.0001)

        # training
        logging = {}

        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = model.score(X_test, y_test)

        # y_pred = model.predict(X_test)
        # logging['precision'] = precision_score(y_test, y_pred, average=None)

        return logging


    def keplerData(self, kepler_df, max_depth):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        # print(X, y)

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        # test_size = 0.25: Means 25% data is in test and 75% in train (you can try to vary it)

        # regressor because output variable is not 0 or 1 but continous
        model = tree.DecisionTreeRegressor(criterion="mse",
                                           splitter="best",
                                           max_depth=max_depth,
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           max_features=None,
                                           max_leaf_nodes=None)

        # training
        logging = {}

        start_time = timeit.default_timer()
        model.fit(X_train, y_train)
        end_time = timeit.default_timer()
        logging['training_time'] = end_time - start_time
        logging['accuracy'] = model.score(X_test, y_test)
        return logging

    def get_results(self):
        print("[*] DT - Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] DT - Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def get_kepler_results(self):
        return self.kepler_graph_data

    def get_insurance_results(self):
        return self.insurance_data

    def __init__(self, runs = 0, variance = "max_depth"):
        print("Decision Tree, using Kepler/Insurance data set")
        print(f"Runs: {runs}. Using {variance} as the treatment.")
        self.variance = variance
        self.kepler_graph_data = pd.DataFrame(columns=['runs', 'accuracy', 'runtime'], index=range(runs))
        self.insurance_data = pd.DataFrame(columns=['runs', 'accuracy', 'runtime'], index=range(runs))

        for i in range(runs):
            if i == 0:
                continue

            kepler_df = exp_runner.get_kepler_train_test_data()
            insurance_df = exp_runner.get_insurance_train_test_data()

            if self.variance == 'max_depth':
                self.kepler_logging = self.keplerData(kepler_df, i)
                self.insurance_logging = self.insuranceData(insurance_df, i)
            else:
                print("Not a valid learning parameter")
                pass

            self.kepler_graph_data.loc[i].runs = i
            self.kepler_graph_data.loc[i].accuracy = self.kepler_logging.get('accuracy')
            self.kepler_graph_data.loc[i].runtime = self.kepler_logging.get('training_time')
            self.kepler_graph_data.loc[i].precision = self.kepler_logging.get('precision')
            print(f"[*][{i}] DT- Kepler Data Accuracy: {self.kepler_logging.get('accuracy')}")
            print(f"[*][{i}] DT- Kepler Training Runtime: {self.kepler_logging.get('training_time')}")

            self.insurance_data.loc[i].runs = i
            self.insurance_data.loc[i].accuracy = self.insurance_logging.get('accuracy')
            self.insurance_data.loc[i].runtime = self.insurance_logging.get('training_time')
            self.insurance_data.loc[i].precision = self.insurance_logging.get('precision')
            print(f"[*][{i}] DT- Insurance Data Accuracy: {self.insurance_logging.get('accuracy')}")
            print(f"[*][{i}] DT- Insurance Data Precision: {self.insurance_logging.get('precision')}")
            print(f"[*][{i}] DT- Insurance Training Runtime: {self.insurance_logging.get('training_time')}")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


