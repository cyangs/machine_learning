1. You used KNeighborsClassifier: But this dataset is not a classification based dataset.

Classfication-> output is discrete (like 0 or 1)
Regression-> output is continuous variable: Like this one

- So, `KNeighborsRegressor` should be used.

2. Same problem with `AdaBoostRegressor`

3. Implemeted the SVM function

4. Same problem with neural network
- since classification report and confusion matrix not work on continous data I commented them out.

5. I have set up function for better model as neural network:
function named as keplerDataBetterModel:
max_iter=1000, will take time to train

Accuracy is so low, because first of all, we removed all of the columns which were not float. 
So all string columns. So ofc the accuracy will get low, also this is just a dataset we don't know if there exists a cor-relation or not, so accuracy may get low as we have no reference frame.

travel_insurance.csv is good.

It also have string columns, where values of columns are in numeric form.
There you have to encode the column strings in to numeric: Like a mapping where YES->1, NO->0: After that all of the models can be used, get it? This is called data-preprocessing of converting categorical data in numerical data, this is a good practice you should look it up, and then ask me if any doubts.

So, 1 big mistake that is making all the difference, which I figured out in the next 5 min is that you are actually trying to classify a discrete number of classes (possible Destinations)
this is a classifier problem
But you are using regression everywhere, just opposite to the last mistake you made.

classification vs regression. 

I, then went ahead and saw that you are training your model using only 4-5 columns (as rest are categorical.)
should i have more
I then converted all the columns into numeric, be it category or else.
Yes.
I used all columns of insurance dataset.

The destination that you are trying to predict is not well-correlated with other features so it can't get past 40 or so.

I tried to predict Age on this dataset: 50% accuracy.
I predicted Gender: 90% accuracy.

But destination labels are too much, but not much data.
Yes.
Ok.
I''ve uploaded the two files that I've changed here.


- I have changed preprocessing of insurance dataset
- And added and changed model in DT file, in insurance function
This classifier problem must be coming in rest of the models,
so you should use classifier in rest of the files.
Too, that would give right results (not accurate but right)


QUESTIONS:
1. SVM - How to test different kernels. rbf works fine, I killed linear. 
2. Implementing cross validation?
3. Check validation accuracy as well
5. Validation curve?
6. How to avoid overfitting?
7. NN Params? layers/activation functions? What to test? 
8. confusion matrix.

we need to show both learning curve and model complexity analysis. That's 5 (algorithms) * 2 (datasets) *3 (learning curve + 2 hyper parameters) = 30 graphs. Do we really need 30 graphs in a 10 page report? 

A hyperparameter is just a parameter whose value is set before the learning process, whereas a parameter's value changes during the training process. By that definition, yes, I would say that learning rate is a hyperparameter for boosting

