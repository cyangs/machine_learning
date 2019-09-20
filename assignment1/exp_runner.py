from assignment1 import DT
from assignment1 import KNN
from assignment1 import SVM
from assignment1 import BOOST
from assignment1 import NN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

def get_kepler_train_test_data():
    df = pd.read_csv("../data/kepler.csv")
    df_copy = df.copy()

    non_floats = []

    koi_disposition = {
        'koi_disposition': {
            'CONFIRMED': 0,
            'FALSE POSITIVE': 1,
            'CANDIDATE': 2
        }
    }

    df_copy.replace(koi_disposition, inplace=True)
    df_copy['koi_disposition'] = df_copy['koi_disposition'].astype(np.float64)

    # drop all columns who are not float
    for col in df_copy:
        if df_copy[col].dtypes != "float64":
            non_floats.append(col)

    df_copy = df_copy.drop(columns=non_floats)

    # replace all NaN with 0
    df_copy = df_copy.fillna(0)

    # X = X._get_numeric_data
    # print(X, y)

    return df_copy

def get_insurance_train_test_data():
    df = pd.read_csv("../data/travel_insurance.csv")
    df_copy = df.copy()

    non_floats = []

    countries = df['Destination'].astype('category').cat.categories.tolist()
    replace_countries = {'Destination': {k: v for k, v in zip(countries, list(range(1, len(countries) + 1)))}}
    df_copy.replace(replace_countries, inplace=True)

    boolean_claim_replace = {'Claim': {'Yes': 0, 'No': 1}}
    # boolean_gender_replace = { 'Gender' : { 'F' : 0, 'M' : 1, 'nan': 2}}

    df_copy.replace(boolean_claim_replace, inplace=True)
    df_copy.drop('Gender', axis=1, inplace=True)

    # drop all columns who are not float
    for col in df_copy:
        if df_copy[col].dtypes != "float64":
            if df_copy[col].dtypes == "int64":
                df_copy[col] = df_copy[col].astype(np.float64)
            else:
                non_floats.append(col)

    df_copy = df_copy.drop(columns=non_floats)
    df_copy = df_copy.fillna(0)

    return df_copy


if __name__ == '__main__':

    print("Initializing Models...")

    # runs the Decision Tree
    # dtRunner = DT.DecisionTree()

    # runs the K-Nearest Neighbor
    kNNRunner = KNN.KNearestNeighbor(50)
    kNNRunner.plot_data()
    #
    # runs the Support Vector Machine
    # SVMRunner = SVM.SupportVectorMachine(kepler_df, insurance_df)
    #
    # runs the Boosting
    # BoostRunner = BOOST.Boosting()

    # runs the Neural Network
    # nnRunner = NN.NeuralNetwork()
    #
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    # dtRunner.get_results()
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    # kNNRunner.get_results()
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    # BoostRunner.get_results()
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    # nnRunner.get_results()
    # print(">>>>>>>>>>>>>>>>>>>>>>")


