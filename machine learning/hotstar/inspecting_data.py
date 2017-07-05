# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from external_functions import printing_Kfold_scores
from sklearn.svm import SVC

test_data = pd.read_csv("test_data.csv")
train_data = pd.read_csv("train_data.csv")

ID = pd.read_csv("ID.csv")

inspect_test = test_data.head()
inspect_train = train_data.head()
 

test_data.drop(['Unnamed: 0'], inplace=True, axis=1)
train_data.drop(['Unnamed: 0'], inplace=True, axis=1)
ID.drop(['Unnamed: 0'], inplace=True, axis=1)


#checking target classes 

count_classes = pd.value_counts(train_data['segment'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

#skewed data 
test_other = test_data[['Family',"Crime",'Kabaddi','Boxing',"Mythology","Reality"]]
train_other=train_data[['Family',"Crime",'Kabaddi','Boxing',"Mythology","Reality"]]


test_data.drop(['Family',"Crime",'Kabaddi','Boxing',"Mythology","Reality"], inplace=True, axis=1)
train_data.drop(['Family',"Crime",'Kabaddi','Boxing',"Mythology","Reality"], inplace=True, axis=1)

test_data = pd.concat([test_other, test_data], axis=1, ignore_index=False)
train_data = pd.concat([train_other, train_data], axis=1, ignore_index=False)


heading_test = list(test_data) #heading for data frame 
heading_train = list(train_data) 
i=0

while i in range(len(heading_test)):
        print(heading_test[i],heading_train[i])
        i =i+1




number_records_one = len(train_data[train_data.segment == 1])
one_indices = np.array(train_data[train_data.segment == 1].index)

# Picking the indices of the normal classes
normal_indices = train_data[train_data.segment == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, int(number_records_one * 1.45), replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([one_indices,random_normal_indices])

# Under sample dataset
under_sample_data = train_data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'segment']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'segment']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.segment == 0])/float(len(under_sample_data)))
print("Percentage of one transactions: ", len(under_sample_data[under_sample_data.segment == 1])/float(len(under_sample_data)))
print("Total number of transactions in resampled data: ", len(under_sample_data))

################################################################################



#prediction part


y = np.array(train_data['segment'])
train_data.drop(['segment'],inplace=True, axis=1)


X = np.array(train_data)

from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))


###############################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.svm import SVC


def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            # Call the logistic regression model with a certain C parameter
            lr =LogisticRegression( C=c_param,penalty='l1',class_weight='balanced')
            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.loc[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c



###############################################################################
best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)


lr = LogisticRegression(C = 0.01, penalty = 'l1')
y_pred_undersample_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)

fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(),y_pred_undersample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

X_submit = np.array(test_data)
y_submit = lr.predict_proba(X_submit)
y_submit[:,1] =  y_submit[:,1] > 0.38
columns = ['segment']
sub = pd.DataFrame(data=y_submit[:,1], columns=columns)
sub['ID'] = ID
sub = sub[['ID','segment']]
sub.to_csv("sub_hot.csv", index=False)

new = lr.coef_
indexes =[]
i=0
len(new[0])
while i in range(len(new[0])):
    if new[0][i] == 0.00:
        indexes.append(i)
    i=i+1
    
drop_array = [ heading_test[i] for i in indexes ]

test_data.drop(drop_array, inplace=True, axis=1)
train_data.drop(drop_array, inplace=True, axis=1)


###############################################################################
