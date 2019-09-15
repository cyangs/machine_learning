import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split

class KNearestNeighbor:

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
        y = df['koi_score']
        X = df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        if self.verbose:
            print(X, y)

        model = neighbors.KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=1
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        model.fit(X_train, y_train)

        # accuracy
        accuracy = model.score(X_test, y_test)
        print("[*] Accuracy: {}".format(accuracy))

    def __init__(self, test = True, verbose = True):
        self.verbose = verbose

        if test:
            return NotImplementedError
        else:
            print("Production Run, using Kepler data set")
            self.keplerData()
