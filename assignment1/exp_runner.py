from assignment1 import DT
from assignment1 import KNN
from assignment1 import SVM
from assignment1 import BOOST
from assignment1 import NN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

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


def plot_boost_model(kep_df, ins_df, variance = 'n_estimators'):
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Data...")
    ax = plt.gca()
    kep_df.plot(kind='line', x='runs', y='accuracy', color='red', ax=ax)
    ins_df.plot(kind='line', x='runs', y='accuracy', color='blue', ax=ax)
    plt.savefig(f'./output/BOOST_graph_{variance}_{timestamp}.png')
    plt.clf()

def plot_kNN_model(kep_df, ins_df, variance = 'n_neighbors'):
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Data...")
    ax = plt.gca()
    kep_df.plot(kind='line', x='runs', y='accuracy', color='red', ax=ax)
    ins_df.plot(kind='line', x='runs', y='accuracy', color='blue', ax=ax)
    plt.savefig(f'./output/KNN_graph_{variance}_{timestamp}.png')
    plt.clf()

def plot_DT_model():

if __name__ == '__main__':
    print("Initializing Models...")

    # runs the Decision Tree
    dtRunner = DT.DecisionTree()
    plot_DT_model()
    #
    # # runs the K-Nearest Neighbor
    kNNRunner = KNN.KNearestNeighbor(50)
    plot_kNN_model(kNNRunner.get_kepler_results(), kNNRunner.get_insurance_results())
    # # #
    # # runs the Support Vector Machine
    # SVMRunner = SVM.SupportVectorMachine()
    # #
    # runs the Boosting
    boostModel = BOOST.Boosting(50)
    plot_boost_model(boostModel.get_kepler_results(), boostModel.get_insurance_results())

    boostModel = BOOST.Boosting(50, "learning_rate")
    plot_boost_model(boostModel.get_kepler_results(), boostModel.get_insurance_results())

    # runs the Neural Network
    # nnRunner = NN.NeuralNetwork()
    #
