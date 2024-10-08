import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

#Reading in data
dataframe = pd.read_csv('C:/Users/Herschelle/Downloads/NFT_Sales.csv')
dataframe = dataframe.fillna(0)
entries, cols = dataframe.shape
print(f'There are {entries} entries and {cols} columns present in the dataset')

#There is no target variable provided, so a very basic one is created for model training purposes
median_num_sales = dataframe['Number_of_Sales'].median()
dataframe['temp_target'] = (dataframe['Number_of_Sales'] > median_num_sales).astype(int)
del dataframe['Date']

#Use Random Forest Classifier to highlight important of respective factors(Important for our algorithm)
model = RandomForestClassifier(random_state=42)
model.fit(dataframe.drop('temp_target', axis=1), dataframe['temp_target'])

#Now the important of respective features can be considered
feature_important = model.feature_importances_

#Put all features into a dataframe(excluding the target column), and sorts them in descending order
features_df = pd.DataFrame({'Feature': dataframe.columns[:-1], 'Important Evaluation': feature_important})
features_df.sort_values(by='Important Evaluation', ascending=False, inplace=True)
#print(features_df.head(3))
#Assign the features to their respective variables(used in algorithm)
w_numsales = feature_important[7]
w_primarysales = feature_important[9]
w_secondarysales = feature_important[4]

#Now that we have gotten our features, the temp target column can be removed
del dataframe['temp_target']

#Create algorithm to decide if NFT is worth investment
dataframe['Score_Target'] = ((dataframe['Number_of_Sales']/(dataframe['Number_of_Sales'].max()))*w_numsales + (dataframe['Primary_Sales']/dataframe['Primary_Sales'].max())*w_primarysales + (dataframe['Secondary_Sales_cumsum']/dataframe['Secondary_Sales_cumsum'].max())*w_secondarysales)

#Now that the target variable has been added and the algorithm has been applied, if the score is >0,5, it will be 1, else 0
dataframe['Score_Target'] = dataframe['Score_Target'].apply(lambda x: 1 if x > 0.005 else 0)

#Extract features and target variable(s)
features = dataframe[['Number_of_Sales', 'Primary_Sales', 'Secondary_Sales_cumsum']]
target = dataframe['Score_Target']

#Split data into training and testing sets, and apply scaling
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#The logistic regression model will now be created and trained, and the predictions will be made
logistic_reg_model = LogisticRegression(random_state = 42)
logistic_reg_model.fit(scaled_X_train, Y_train)
Y_predictions = logistic_reg_model.predict(scaled_X_test)

#We can now evaluate the accuracy of the model
model_accuracy = accuracy_score(Y_test, Y_predictions)
print(f'The model had an accuracy score of {model_accuracy}')

#The confusion matrix will be evaluated as well
model_confusion_matrix = confusion_matrix(Y_test, Y_predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(model_confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Sales', 'High Sales'],
            yticklabels=['Low Sales', 'High Sales'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Lastly, the classification report will be provided as well
model_cf = classification_report(Y_test, Y_predictions)
print('Classification Report: \n')
print(model_cf)



