# Home-Credit-Default-Risk
Abstract
The aim of this project is to use machine learning methodologies on historical loan application data to predict whether or not an applicant will be able to repay a loan. As an extension to the Visual EDA driven feature sampling and baseline model development, the focus for this phase included data modelling to combine available datasets, feature engineering considering categorical and numerical features and implementing experiments using Logistic Regression, Random Forest, Baiyes Naive, XGBoost and Neural Networks. Our results in this phase show that the best performing neural network was multi layer model with just a single layer in between the input and output layers with Relu as the activation function and 256 hidden neurons. The neural network gave a Kaggle AUC score of 74.68%. Overall, XGBoost was the best performing model with validation accuracy 70.49% and 75.41% as the test ROC_AUC,respectively for a balanced dataset. The lowest performing was LR model at 69.31% and 74.64% validation and test AUC(Area under ROC). Our best score in Kaggle submission out of all four submission was 0.74464.




Feature Engineering and transformers
In our feature engineering process, we created a few features within the secondary tables bureau and credit card balance tables by performing certain numerical operations. After that, we aggregated the numerical columns by taking the mean of all the columns by grouping on the SK_ID_CURR column which was then merged with the main applications_train table.

We engineered features which calculated the max, min and mean of all numerical columns in both the primary and secondary tables. Apart from that, we also created new features by doing numerical calculations on the columns of the secondary tables such as the bureau table, cc balance table, installments tables, etc. Some examples of new features are:

BUREAU_CREDIT_TO_ANNUITY_RATIO (Bureau Table) - Based on the ratio of bureau amount credit sum and bureau amount annuity
PREVAPP_APPLICATION_CREDIT_DIFF (PrevApps Table) - Based on the ratio prevapp amt application and prevapp amount credit
CC_BAL_LIMIT_USE (CC_bal Table) - Based on the ratio of cc balance amount balance and cc balance amount credit limit actual
The way we went about doing this was first by doing OHE to convert categorical columns to numeric columns. After that we grouped the columns on SK_ID by taking the mean of the grouping. Next, we added the mean, max and min columns to all the secondary tables followed by adding new features to the tables.

After the aggregation was performed followed by every individual merging of the secondary tables with the primary table, we generated the correlation of all the features with respect to the ‘TARGET’ variable and based on the absolute correlation values, we selected the top 30 features which were the final numerical features for the base pipeline. These were then merged with the categorical features and in total our baseline model consisted of 37 features ( 30 numerical and 7 categorical)


Pipelines
Phase - 1
Logistic regression model was used as a baseline Model. Training a logistic regression model doesn't require high computation power. Numerical pipeline was used which included the steps of Feature Selection, Imputation and Standardization. Categorical pipeline was used which included Standardization and one-hot encoding. Implementing Logistic Regression as a baseline model is a good starting point for classification tasks due to its easy implementation and low computational requirements. We prepared our data pipeline of Logistic Regression with default parameters (penalty = 'l2', C = 1.0, solver = 'lbfgs', tol = 1e-4).

Here is the high-level workflow for the model pipeline followed by detailed steps:

Downloaded data and performed data pre-processing tasks (joining primary and secondary datasets, transformation)
Created a data pipeline with highly correlated numerical and categorical features.
Imputed missing numerical attribute with mean values and categorical values with the ‘missing’ word
Applied FeatureUnion to combine both Numerical and Categorical features.
Created model with data pipeline and baseline model to fit training dataset
Evaluated the model using accuracy score, AUC score, RMSE and MAE for train, validation and test datasets. Recorded the results in a dataframe.

Phase - 2
In the second phase, apart from logistic regression we experimented by training different models and checking out the test AUC. After the extensive feature engineering process, we decided to do hyperparameter tuning for different models that we were trying to test. We chose 6 models Logistic Regression, Naive Bayes, Random Forest, XGBoost, KNN and SVM. Of these, it was taking too long to do the hyperparameter tuning of KNN and SVM and so we decided to drop the testing of these models. We then proceeded with the remaining 4 models i.e Logistic Regresison, Naive Bayes, Random Forest and XGBoost. Apart from that, since the dataset was imbalanced, we also sampled the data to get accurate results. Once we got the best parameters for all the 4 models based on the previous steps, we trained the models on those paraemeters to finally get the test results for the all the models. We found that XGBoost was showing the best performance from among all the models.

Phase - 3
In the third phase, we implemented the following combinations of neural networks in our quest to better our AUC score:

Single Layer Neural Network
Multilayer Neural Network:
  Single Hidden Layer
  Double Hidden Layers
Within these neural netowrks, we also did hyperparamter tuning for the number of neurons as well as the learning rate. Binary cross entropy loss(CXE) was used as the evaluation metric for the hyperparameter tuning of the neural networks. We submitted our test values on Kaggle and noted down the Kaggle AUC scores as well. Based on that, we found that a multilayer neural network with a single hidden layer, 256 hidden neurons and a learning rate of 1e-4 was the best perforoming neural network which gave the Kaggle AUC score of 74.68%. However, we found that the XGBoost model that we developed in phase 2 was still the best performing model for this project.

Discussion of Results:
Based on the models discussed above, XGBoost stood out as the best predictive model using the top 50 features. Please note that we are noting the training accuracy based on the results we obtained on the balanced dataset:

Logistic Regression: This model was chosen as the baseline model trained. The training accuracy for this model was 68.34% and test accuracy was 69.33%. A 74.64% AUC score resulted with best parameters for this model. We used the log loss function for this model.

Naive Bayes: Training accuracy of 62.32% and test accuracy of 79.36% was achieved in this model. Test AUC under the curve for this model came out to 70.44%.

Random Forest: The accuracy of the training and test are 73.60% and test 72.17%. Test ROC under the curve is 74.35%. We used the Gini impurity loss function forn this model.

XGBoost: By far this model resulted in the best model. The accuracy of the training and test are 70.93% and test 70.43%. Test ROC under the curve is 75.41% which was the highest. We used the binary logistic loss function for this model.

Neural Networks: We got Kaggle AUC score of 74.68% for a multilayer neural network with a single hidden layer, 256 neurons and learning rate of 1e-4. It was our second best model for this dataset.






