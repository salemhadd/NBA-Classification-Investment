# NBA Player Investment Classifier

**Overview**

This project aims to build a classifier that predicts whether an NBA player is a good investment, i.e., whether they are likely to have a career lasting more than 5 years in the NBA. The classifier is trained using data from the `nba_logreg.csv` dataset.

**Repository Structure**

- `model.py`: Contains the code for reading and decoding the dataset, training the classifier, and scoring it based on recall.
- `app.py`: This is the Flask application for deploying the trained model as a web service.
- `model.pkl`: The serialized pre-trained model.


**Dependencies**

- Python 3.x
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - flask (for the web service)
  - numpy


**Data Exploration and Preprocessing**

The project begins with data exploration, where we analyze the dataset and preprocess it for model training. Key steps include:

1. Identifying available features in the dataset.
2. Describing statistical information about the data.
3. Exploring the distribution of the target variable (`TARGET_5Yrs`).
4. Handling missing values (replacing NaN values).
5. Normalizing the dataset using Min-Max scaling.

**Model Training and Evaluation**

The model training and evaluation are performed as follows:

- We consider various machine learning classifiers, including Support Vector Classifier (SVC), Random Forest Classifier, Logistic Regression, Gaussian Naive Bayes, Decision Tree Classifier, and K-Nearest Neighbors Classifier.
- We use a 3-fold cross-validation strategy to train and test the models, assessing their performance using recall score.
- We choose the best-performing model, SVC, for further analysis.

**Grid Search for SVC**

We conduct a grid search for the Support Vector Classifier (SVC) to determine the optimal combination of hyperparameters. The grid search explores different regularization parameters (C) and kernel types (linear, poly, rbf, sigmoid) to find the best-performing SVC model.

**Flask Web Service Integration**

The trained SVC model is integrated into a Flask web service (`model.py`) to make predictions. Users can input the player's statistics, and the service provides predictions on whether it's worth investing in the player.

**Usage**

- Ensure the required dependencies are installed.
- Run `model.py` to train and evaluate the model.
- Run `app.py` to start the Flask web service.

**Conclusion**

This project provides a solution to predict the investment potential of NBA players based on their sports statistics. It involves data exploration, model training, and integration into a web service. The choice of model, hyperparameters, and scoring function is critical to achieving the best results.

For any questions or clarifications, please contact [Salem Haddad / salem.haddadbecha@gmail.com].

