import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split

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

    def keplerData(self):
        # read csv file | df is dataframe (object of pandas module)
        df = pd.read_csv("../data/kepler.csv")
        non_floats = []

        # drop all columns who are not float
        for col in df:
            if df[col].dtypes != "float64":
                non_floats.append(col)
        df = df.drop(columns=non_floats)

        # replace all NaN with 0
        df = df.fillna(0)

        # koi_score is output variables (that we need to predict)
        y = df['koi_score']
        X = df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        print(X, y)

        # divide X and y into train and test | train on different data | test on different -> good matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        # test_size = 0.25: Means 25% data is in test and 75% in train (you can try to vary it)

        # regressor because output variable is not 0 or 1 but continous
        model = tree.DecisionTreeRegressor(criterion="mse",
                                           splitter="best",
                                           max_depth=None,
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           max_features=None,
                                           max_leaf_nodes=None)

        # training
        model.fit(X_train, y_train)

        # accuracy
        accuracy = model.score(X_test, y_test)
        print("[*] Accuracy: {}".format(accuracy))

    def __init__(self, test = True):
        if test:
            print("Test run, using default Iris data set")
            self.testRunWithIrisData()
        else:
            print("Production Run, using Kepler data set")
            self.keplerData()

