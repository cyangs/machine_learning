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

# def get_insurance_train_test_data():
#     df = pd.read_csv("../data/travel_insurance.csv")
#     df_copy = df.copy()
#
#     non_floats = []
#
#     countries = df['Destination'].astype('category').cat.categories.tolist()
#     replace_countries = {'Destination': {k: v for k, v in zip(countries, list(range(1, len(countries) + 1)))}}
#     df_copy.replace(replace_countries, inplace=True)
#
#     boolean_claim_replace = {'Claim': {'Yes': 0, 'No': 1}}
#     # boolean_gender_replace = { 'Gender' : { 'F' : 0, 'M' : 1, 'nan': 2}}
#
#     df_copy.replace(boolean_claim_replace, inplace=True)
#     df_copy.drop('Gender', axis=1, inplace=True)
#
#     # drop all columns who are not float
#     for col in df_copy:
#         if df_copy[col].dtypes != "float64":
#             if df_copy[col].dtypes == "int64":
#                 df_copy[col] = df_copy[col].astype(np.float64)
#             else:
#                 non_floats.append(col)
#
#     df_copy = df_copy.drop(columns=non_floats)
#     df_copy = df_copy.fillna(0)
#
#     return df_copy


def get_insurance_train_test_data():
    df_copy = pd.read_csv("../data/travel_insurance.csv")

    col_list = df_copy.columns
    for col in col_list:
        if df_copy[col].dtype == 'O':
            df_copy[col] = df_copy[col].astype('category').cat.codes

    df_copy = df_copy.fillna(0)

    return df_copy


def plot_boost_model(kep_df, ins_df, variance = 'n_estimators', runtime = True):
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Data...")
    ax = plt.gca()
    plt.title(f"Boosting ({variance} vs Accuracy)")
    plt.xlabel(f"Number of {variance}")
    plt.ylabel("Accuracy")
    kep_df.plot(kind='line', x='estimators', y='accuracy', color='red', ax=ax, label="Kepler")
    ins_df.plot(kind='line', x='estimators', y='accuracy', color='blue', ax=ax, label="Insurance")
    plt.savefig(f'./output/BOOST_graph_{variance}_{timestamp}.png')
    plt.clf()

    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Runtime Data...")
    ax = plt.gca()
    plt.title("Boosting (n Estimators vs Runtime)")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Runtime (sec)")
    kep_df.plot(kind='line', x='estimators', y='runtime', color='red', ax=ax, label="Kepler")
    ins_df.plot(kind='line', x='estimators', y='runtime', color='blue', ax=ax, label="Insurance")
    plt.savefig(f'./output/BOOST_graph_RUNTIME_{timestamp}.png')
    plt.clf()

def plot_kNN_model(kep_df, ins_df, variance = 'n_neighbors'):
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Accuracy Data...")
    ax = plt.gca()
    plt.title("k-Nearest Neighbor (k Neighbors vs Runtime)")
    plt.xlabel("k Neighbors")
    plt.ylabel("Accuracy")
    kep_df.plot(kind='line', x='neighbors', y='accuracy', color='red', ax=ax, label="Kepler")
    ins_df.plot(kind='line', x='neighbors', y='accuracy', color='blue', ax=ax, label="Insurance")
    plt.savefig(f'./output/KNN_graph_{variance}_{timestamp}.png')
    plt.clf()

    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Runtime Data...")
    ax = plt.gca()
    plt.title("k-Nearest Neighbor (k Neighbors vs Runtime)")
    plt.xlabel("k Neighbors")
    plt.ylabel("Runtime (sec)")
    kep_df.plot(kind='line', x='neighbors', y='runtime', color='red', ax=ax, label="Kepler")
    ins_df.plot(kind='line', x='neighbors', y='runtime', color='blue', ax=ax, label="Insurance")
    plt.savefig(f'./output/KNN_graph_RUNTIME_{timestamp}.png')
    plt.clf()


def plot_DT_model(kep_df, ins_df, variance = 'max_depth'):
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Data...")
    ax = plt.gca()
    plt.title("Decision Tree (Max Depth vs. Accuracy)")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    kep_df.plot(kind='line', x='max_depth', y='accuracy', color='red', ax=ax, label="Kepler")
    ins_df.plot(kind='line', x='max_depth', y='accuracy', color='blue', ax=ax, label="Insurance")
    plt.savefig(f'./output/DT_graph_{variance}_{timestamp}.png')
    plt.clf()

    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Runtime Data...")
    ax = plt.gca()
    plt.title("Decision Tree (Max Depth vs. Runtime)")
    plt.xlabel(variance)
    plt.ylabel("Runtime (sec)")
    kep_df.plot(kind='line', x='max_depth', y='runtime', color='red', ax=ax, label="Kepler")
    ins_df.plot(kind='line', x='max_depth', y='runtime', color='blue', ax=ax, label="Insurance")
    plt.savefig(f'./output/DT_graph_RUNTIME_{timestamp}.png')
    plt.clf()


def plot_SVM_model(df, dataname):
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Plotting Data...")

    plt.title("Support Vector Machine")
    df.plot.bar(rot=0)
    plt.xlabel("Kernels")
    plt.ylabel("Accuracy")
    plt.savefig(f'./output/SVM_Graph_{timestamp}.png')
    plt.clf()


if __name__ == '__main__':
    # user_input = input("Run which model?")
    print("Initializing Models...")

    # runs the Decision Tree
    # dtRunner = DT.DecisionTree(40)
    # plot_DT_model(dtRunner.get_kepler_results(), dtRunner.get_insurance_results())
    # #
    # # # runs the K-Nearest Neighbor
    # kNNRunner = KNN.KNearestNeighbor(70)
    # plot_kNN_model(kNNRunner.get_kepler_results(), kNNRunner.get_insurance_results())
    # # # #
    # runs the Support Vector Machine
    # SVMRunner = SVM.SupportVectorMachine()
    # plot_SVM_model(SVMRunner.get_kepler_results(), 'Kepler')
    # plot_SVM_model(SVMRunner.get_insurance_results(), 'Insurance')

    # # # #
    # # # runs the Boosting
    # boostModel = BOOST.Boosting(300)
    # plot_boost_model(boostModel.get_kepler_results(), boostModel.get_insurance_results())
    #
    boostModel = BOOST.Boosting(50, 'learning_rate')
    plot_boost_model(boostModel.get_kepler_results(), boostModel.get_insurance_results(), 'learning_rate', False)


    # runs the Neural Network
    # nnRunner = NN.NeuralNetwork()
    #

    # [Parallel(n_jobs=2)]: Done 336 out of 336 | elapsed: 69.0min finished
    # [*] Learned parameters: {'MLP__activation': 'identity', 'MLP__alpha': 4.0, 'MLP__hidden_layer_sizes': (64,), 'MLP__random_state': 1, 'MLP__solver': 'lbfgs'}
    # [*] NN - Kepler Data Accuracy: 0.028935758890765717
    # [*] NN - Kepler Training Time: 737.944311378
    # [*] NN - Insurance Data Accuracy: 0.032214468286650644
    # [*] NN - Insurance Training Time: 4174.011079383