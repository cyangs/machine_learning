
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

class NeuralNetwork:

    def testRun(self):
        return NotImplementedError

    def keplerData(self):
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

        mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
        mlp.fit(X_train, y_train)

        predictions = mlp.predict(X_test)

        print(confusion_matrix(y_test, predictions))

        print(classification_report(y_test, predictions))

    def __init__(self, test=True, verbose=True):
        self.verbose = verbose

        if test:
            return NotImplementedError
        else:
            print("Production Run, using Kepler data set")
            self.keplerData()