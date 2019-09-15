import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn import metrics

class Boosting:

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
        # 70% training and 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        classifier = ensemble.AdaBoostClassifier(
            base_estimator=None,
            n_estimators=50,
            learning_rate=1,
            algorithm='SAMME.R',
            random_state=None)

        boost_model = classifier.fit(X_train, y_train)
        y_pred = boost_model.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    def __init__(self, test=True, verbose=True):
        self.verbose = verbose

        if test:
            return NotImplementedError
        else:
            print("Production Run, using Kepler data set")
            self.keplerData()