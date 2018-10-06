
# coding: utf-8

# In[1]:


# Input Data

import numpy as np
import pandas as pd
train_set =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels

# Check the Value that need to be predicted is a Categorical or Continuous
train_set.head()
print(train_set.wage_class.unique())


# In[2]:


train_set.head(100)


# In[3]:


test_set.head(100)


# In[4]:


train_set['education'].value_counts() # Catgorical data


# In[5]:


train_set['education_num'].value_counts() # same as education, already categorized


# In[6]:


train_set['workclass'].value_counts() # Catgorical data


# In[7]:


# Encode all catgerical values using factorize

train = train_set.copy(deep=True)

char_cols = train.dtypes.pipe(lambda x: x[x == 'object']).index # Get the Columns which is of type Object for which categorical value to be converted

#for c in char_cols:
 #   train[c] = pd.factorize(train[c])[0]

#Here label_mapping will be having all unique values for each column stored in a dictionary
label_mapping= {}
for c in char_cols:
    train[c], label_mapping[c] = pd.factorize(train[c]) 
    

test = test_set.copy(deep=True)

for c in char_cols:
    test[c] = pd.factorize(test[c])[0]


test.head(50)


# In[8]:


#Create Features and Labels
colstodrop={'education','wage_class'} # Deleting 'education' as education_num column is already encoded in raw data
X_train = train.drop(colstodrop, axis=1)
y_train = train.wage_class
X_test = test.drop(colstodrop, axis=1)
y_test = test.wage_class


# In[12]:


# Import required packages for XGBOOST
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier


# In[14]:


# Add Parameter for XGBOOST classifiers
params = {
'objective': 'binary:logistic',
'max_depth': 2,
'learning_rate': 1.0,
'silent': 1.0,
'n_estimators': 5
}

#Train the Data
bst = XGBClassifier(**params).fit(X_train, y_train)

# Predict the training and test data using the models
y_train_preds = bst.predict(X_train)
y_test_preds = bst.predict(X_test)


# In[15]:


#Method to calculate the accuracy of the Prediction

def calculateaccuracy(ytest, y_preds, predType):
    '''
    Method to calculate the accuracy of the Prediction
    '''
    correct = 0
    for i in range(len(y_preds)):
        if (y_preds[i] == ytest[i]):
            correct += 1

    acc = accuracy_score(ytest, y_preds)
    print(predType + ' Data Predicted correctly: {0}/{1}'.format(correct, len(y_preds)))
    print('Accuracy: {0:.2f}'.format(acc))
    print('Error: {0:.4f}'.format(1-acc))


# In[16]:


# Calculate Prediction Accuracy for TRAINED DATA
calculateaccuracy(y_train, y_train_preds, "TRAIN")


# In[17]:


# Calculate Prediction Accuracy for TEST DATA
calculateaccuracy(y_test, y_test_preds, "TEST")

