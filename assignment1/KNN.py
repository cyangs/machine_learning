from sklearn import neighbors
from sklearn.model_selection import train_test_split
from assignment1 import exp_runner

class KNearestNeighbor:

    def testRun(self):
        return NotImplementedError

    def insuranceData(self, insurance_df):
        # Destination is output variables (that we need to predict)
        y = insurance_df['Destination']
        X = insurance_df
        del X['Destination']  # delete from X we don't need it

        # X = X._get_numeric_data
        print(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        model = neighbors.KNeighborsRegressor(
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
        return model.score(X_test, y_test)


    def keplerData(self, kepler_df):
        # koi_score is output variables (that we need to predict)
        y = kepler_df['koi_score']
        X = kepler_df
        del X['koi_score']  # delete from X we don't need it

        # X = X._get_numeric_data
        # if self.verbose:
        #     print(X, y)

        model = neighbors.KNeighborsRegressor(
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
        return model.score(X_test, y_test)

    def get_results(self):
        print("[*] KNN- Kepler Data Accuracy: {}".format(self.kepler_accuracy))
        print("[*] KNN- Insurance Data Accuracy: {}".format(self.insurance_accuracy))

    def __init__(self):
        print("k-nearest neighbor, using Kepler/Insurance data set")
        self.kepler_accuracy = self.keplerData(exp_runner.get_kepler_train_test_data())
        self.insurance_accuracy = self.insuranceData(exp_runner.get_insurance_train_test_data())
        self.get_results()

