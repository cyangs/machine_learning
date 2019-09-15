import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


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
            out_file="myTreeName.dot",
            feature_names=list(X.columns),
            class_names=iris.target_names,
            filled=True,
            rounded=True)

        sample_one_pred = int(tree.predict([[5, 5, 1, 3]]))
        sample_two_pred = int(tree.predict([[5, 5, 2.6, 1.5]]))
        print(f"The first sample most likely belongs a {iris.target_names[sample_one_pred]} flower.")
        print(f"The second sample most likely belongs a {iris.target_names[sample_two_pred]} flower.")

    def productionRunWithOkCupidData(self):

        print("Unitialized yet. ")




    def __init__(self, test = False):
        if test:
            print("Test run, using default Iris data set")
            self.testRunWithIrisData()
        else:
            print("Production Run, using OkCupid data set")
            self.productionRunWithOkCupidData()

runnable = DecisionTree(True)
