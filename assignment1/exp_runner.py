from assignment1 import DT
from assignment1 import KNN
from assignment1 import SVM
from assignment1 import BOOST
from assignment1 import NN

if __name__ == '__main__':

    print("Initializing Models")
    
    # First param is Test mode, Default is False for Actual Data Run
    # Second param is Verbose, Default is True for output

    # runs the Decision Tree
    dtRunner = DT.DecisionTree(False, True)

    # runs the K-Nearest Neighbor
    kNNRunner = KNN.KNearestNeighbor(False, True)

    # runs the Support Vector Machine
    SVMRunner = SVM.SupportVectorMachine(False, True)

    # runs the Boosting
    BoostRunner = BOOST.Boosting(False, True)

    # runs the Neural Network
    nnRunner = NN.NeuralNetwork(False, True)