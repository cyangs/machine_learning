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