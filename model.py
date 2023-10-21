# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.svm import SVC
import itertools
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pickle


def score_classifier(dataset, classifier, labels):
    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3, random_state=50, shuffle=True)
    confusion_mat = np.zeros((2, 2))
    recall = 0

    for training_ids, test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set, training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat += confusion_matrix(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)

    recall /= 3

    return recall, confusion_mat


def data_exploration(df):
    print(df.columns.values)  # Which features are available in the dataset?
    print(df.describe())
    print(df['TARGET_5Yrs'].value_counts())
    print(df.isnull().sum().sum())


def correlation(df):
    df = df.drop(['Name'], axis=1)
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def data_preprocessing(df):
    """
     Visualize correlation Between different labels of the data set
    When running the script, the correlation plot will be displayed
    You have to close it to continue the execution
    """
    # extract names, labels, features names, and values
    labels = df['TARGET_5Yrs'].values
    df_vals = df.drop(['TARGET_5Yrs', 'Name', 'FG%', '3P%', 'FT%'], axis=1).values

    # Replace NaN values
    for x in np.argwhere(np.isnan(df_vals)):
        df_vals[x] = 0.0
    # Normalize dataset
    X = MinMaxScaler().fit_transform(df_vals)

    return X, labels


def grid_search_SVC(X_train, labels):
    """
    Perform a grid search to find the best Support Vector Classifier (SVC) model.
    :param X_train: The training dataset.
    :param labels: The labels used for training and validation.
    :return: A tuple containing information about the best model and its performance.
    """
    # Define a list of regularization parameters and kernel types to perform grid search
    svc_reguls = [0.01, 0.1, 1.0, 10]
    svc_kernels = ["linear", "poly", "rbf", "sigmoid"]
    grid_search_svc = {}  # Initialize an empty dictionary to store results

    # Iterate over all combinations of regularization and kernel
    for regul, kernel in itertools.product(svc_reguls, svc_kernels):
        # Create a Support Vector Classifier with the current parameters
        current_model = SVC(C=regul, kernel=kernel)
        print(current_model)
        str_model = str(current_model)  # Convert the model to a string for identification

        # Call the 'score_classifier' function to evaluate the model
        recall, confusion = score_classifier(dataset=X_train, classifier=current_model, labels=labels)

        # Store the results (recall and confusion matrix) in the 'grid_search_svc' dictionary
        grid_search_svc[(str_model, regul, kernel)] = {'recall': recall, 'confusion': confusion}

    # Sort the results by recall in descending order
    sorted_grid_svc_recall = sorted(grid_search_svc.items(), key=lambda x: x[1]['recall'], reverse=True)

    # Get the best model (the one with the highest recall)
    best_model = sorted_grid_svc_recall[0][0]  # Store the best model string

    # Serialize and save the best model, training data, and labels to a file
    with open('model.pkl', 'wb') as model_file:
        pickle.dump((best_model, X_train, labels), model_file)

        # Return the information about the best model and its performance
        return sorted_grid_svc_recall[0]


def modeling_comparing():
    """
    this function use score_classifier function with several model and compare recall_score.

    :return:
    """
    model1_recall, confusion_matri_1 = score_classifier(X, SVC(), labels)
    model2_recall, confusion_matri_2 = score_classifier(X, RandomForestClassifier(), labels)
    model3_recall, confusion_matri_3 = score_classifier(X, LogisticRegression(), labels)
    model4_recall, confusion_matri_4 = score_classifier(X, GaussianNB(), labels)
    model5_recall, confusion_matri_5 = score_classifier(X, DecisionTreeClassifier(), labels)
    model6_recall, confusion_matri_6 = score_classifier(X, KNeighborsClassifier(), labels)
    print(f'recall score for SVC is  {model1_recall} and  confusion is {confusion_matri_1} ')
    print(f'recall score for RandomForestClassifer is  {model2_recall} and  confusion is {confusion_matri_2} ')
    print(f'recall score for LogisticRegression is  {model3_recall} and matrice confusion is {confusion_matri_3} ')
    print(f'recall score for GaussianNB is  {model4_recall} and  confusion is {confusion_matri_4} ')
    print(f'recall score for DecisionTreeClassifier is  {model5_recall} and  confusion is {confusion_matri_5} ')
    print(f'recall score for KNeighborsClassifier is  {model6_recall} and  confusion is {confusion_matri_6} ')


if __name__ == "__main__":
    df = pd.read_csv("nba_logreg.csv")
    # Data Exploration
    data_exploration(df)
    # Correlation between labels
    correlation(df)
    # Data Preprocessing
    X, labels = data_preprocessing(df)

    # Modeling: Comparing between models
    modeling_comparing()
    # SVC is one of the best model for this problem, we will keep and perform a Grid_search to determine the best param
    best_SVC = grid_search_SVC(X, labels)

    print(f"SVC grid search: {str(best_SVC[0])} best result for recall score but not for confusion {str(best_SVC[1])}")
    # model = SVC(C=best_SVC[0][1], kernel=best_SVC[0][2])
    """
    The recall score is 1.0, which means that the model correctly predicted all instances 
    of the positive class. However, the confusion matrix shows that the model misclassified
     all instances of the negative class (509 of them) as positive, resulting in a high number 
     of false positives.
     In this context, a recall score of 1.0 is not necessarily a good result, especially when
      it comes at the cost of missclassifying all instances of the negative class.
      The best C and kernel parameters for the Support Vector Classifier (SVC) based on the given results 
      are those that maximize the recall score while still providing a reasonable balance between true positives and false positives. I
     ==> We see the log of grid_search to take the best one.
    """
    model1 = SVC(C=0.1, kernel='sigmoid')

    # Fit SVC instance on training data
    #
    model1.fit(X, labels)
    with open('model.pkl', 'wb') as model_file:
        # Serialize and save the 'model' to 'model_file'
        pickle.dump(model1, model_file)
