# Bank Churn Project

This dataset is a bank data at a point,
which has 14 columns and 10000 entries.

## Importing Important Libraries for visualization.
``` python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
## Important Libraries for Mathematical Operations.
```python
import pandas as pd
import numpy as np
```

### Function for generating report from all the ML models
we will use a dataframe with two columns
1. Models  - This column contains the name of the model.
2. Accuracy - This column will contain the accuracy of the model.

```python
Accuracy_Report = pd.DataFrame(columns=["Models","Accuracy"])
models_lis, acc_lis = [], []
def Submit_Score(lis1,lis2):
    models_lis.append(lis1)
    acc_lis.append(lis2)
    return
 ```
 
### Function for showing the final report.
```python
def Show_Model_Score():
    temp_df = pd.DataFrame({'Models': models_lis, 'Accuracy': acc_lis})
    return temp_df
```

## Function for plotting the confusion matrix
Here we will define a function for plotting the confusio matrix
Inputs - Confusion Matrix , Target Names, Color Mapping, Title, Accuracy.

Processing - A plot of confusion matrix.

Output - ![alt text](https://github.com/[username]/[reponame]/blob/master/image.jpg?raw=true) 

```python
def plot_Confusion_matrix(cm, target_names, cmap, title, accuracy):
    
    Submit_Score(title,accuracy)
    
    if cmap is None:
            cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    for i in range(2):
        for j in range(2):
            text = plt.text(j, i, cm[i, j],
                           ha="center", va="center", color="black")


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
    plt.show()
```
```python
#Loading the Churn Data.

churn = pd.read_csv(r"..\dataset\churn.csv")

#Getting Data information.
churn.info()

#Getting some overview of different operations for individual columns
churn.describe()


#Getting some insight of our data.
churn.head()
```

## Plotting BarChart between:-

1. Geography, Exited.
2. Gender, Exited.
3. IsActiveMember, Exited.
4. HasCrCard, Exited.

```python
fig, axarr = plt.subplots(2,2, figsize= (20,30))
sns.countplot(x='Geography', hue='Exited', data=churn, ax= axarr[0,0])

sns.countplot(x='Gender', hue='Exited', data=churn, ax= axarr[0,1])

sns.countplot(x='IsActiveMember', hue='Exited', data=churn, ax= axarr[1,0])

sns.countplot(x='HasCrCard', hue='Exited', data=churn, ax= axarr[1,1])

```
### Observations from above graphs:-
1. Most of the customers were from France who stayed.
2. We lose customers from Germany Usually.
3. Customers who stayed are mostly male.
4. Customers who were not active members have more chances to leave.
5. Usually customers who have Credit card have more chances to stay but its not necessary as over 1100 customers have left who had a credit card.

### Percentage of staying and retaining customers through pie chart.

labels = 'Exited', 'Stayed'
sizes = [churn.Exited[churn['Exited']== 1].count(), churn.Exited[churn['Exited']==0].count()]

explode= (0,0.1)
fig1, ax1= plt.subplots(figsize= (10,8))
ax1.pie(sizes,explode=explode, labels=labels, autopct= '%1.1f%%', shadow= True, startangle=90)

ax1.axis('equal')
plt.title("Proportion of customer Exited and Stayed", size=20)
plt.show()

#### Observations
1. 20.4% persons have retained their bank account.
2. 79.6% have stayed.

### Percentage of Male and Female customers through pie chart.

labels = 'Male', 'Female'
sizes = [churn.Gender[churn['Gender']== 'Male'].count(), churn.Gender[churn['Gender']=='Female'].count()]

explode= (0,0.1)
plt.figure(0)
plt.pie(sizes,explode=explode, labels=labels, autopct= '%1.1f%%', shadow= True, startangle=90)
plt.title("Male and Female Proportion")




labels = 'Yes', 'No'
sizes = [churn.HasCrCard[churn['HasCrCard']== 1].count(), churn.HasCrCard[churn['HasCrCard']==0].count()]

plt.figure(1)
explode=(0,0.1)
plt.pie(sizes,explode=explode, labels=labels, autopct= '%1.1f%%', shadow= True, startangle=90)
plt.title("Credit card holder proportion")


labels = churn['Geography'].unique()
sizes = [churn.Geography[churn['Geography']== 'France'].count(), churn.Geography[churn['Geography']== 'Spain'].count(), churn.Geography[churn['Geography']== 'Germany'].count()]

plt.figure(2)
explode= (0,0.1,0.1)
plt.pie(sizes,explode=explode, labels=labels, autopct= '%1.1f%%', shadow= True, startangle=90)
plt.title("Customer living country proportion")
plt.show()

### Relation Visualzation

churn['Exited_str'] = churn['Exited']
churn['Exited_str'] = churn['Exited_str'].map({1: 'Exited', 0: 'Stayed'})
churn['Exited_str']

!pip install pySankey

from pySankey import sankey


colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    'France':'#f3f71b',
    'Spain':'#12e23f',
    'Germany':'#f78c1b'
}

sankey.sankey(
    churn['Geography'], churn['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12
)



sankey.sankey(
    churn['HasCrCard'], churn['Exited_str'], aspect=20,
    fontsize=12
)
plt.title("Have credit card")
plt.show()


sankey.sankey(
    churn['NumOfProducts'], churn['Exited_str'], aspect=20,
    fontsize=12
)
plt.title("Number of products have")
plt.show()

figure = plt.figure(figsize=(15,8))
plt.hist([
        churn[(churn.Exited==0)]['Age'],
        churn[(churn.Exited==1)]['Age']
        ], 
         
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
plt.xlabel('Age (years)')
plt.ylabel('Number of customers')
plt.legend()

fig, axes = plt.subplots(nrows=4, figsize = (15,30))
fig.subplots_adjust(left=0.2, wspace=0.6)
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist([
        churn[(churn.Exited==0)]['CreditScore'],
        churn[(churn.Exited==1)]['CreditScore']
        ], 
         
    bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax0.legend()
ax0.set_title('Credit Score')

ax1.hist([
        churn[(churn.Exited==0)]['Tenure'],
        churn[(churn.Exited==1)]['Tenure']
        ], 
         
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax1.legend()
ax1.set_title('Tenure')

ax2.hist([
        churn[(churn.Exited==0)]['Balance'],
        churn[(churn.Exited==1)]['Balance']
        ], 
        
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax2.legend()
ax2.set_title('Balance')

ax3.hist([
        churn[(churn.Exited==0)]['EstimatedSalary'],
        churn[(churn.Exited==1)]['EstimatedSalary']
        ], 
         
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax3.legend()
ax3.set_title('Estimated Salary')

fig.tight_layout()
plt.show()

### Final observations 
1. Persons who have 0 balance have less chances to leave.
2. Persons with more credit score have more chances to stay.
3. Females have more chances to leave than males.
4. Most of the customers were from France who stayed.
5. We lose customers from Germany Usually.
6. Customers who stayed are mostly male.
7. Customers who were not active members have more chances to leave.
8. Usually customers who have Credit card have more chances to stay but its not necessary as over 100 customers have left who had a credit card.
9. There is a higher rate of exited clients in Germany and lower in Spain and France.
10. On age, customer below 40 and above 65 years old have a tendency to keep their account.
11. Non active members tend to discontinue their services with a bank compared with the active clients. 
12. The dataset has 96% of clients  with 1 or 2 product, and customers with 1 product only have a higher rate to close the account than those with 2 products (around 3x higher).
13. Estimated Salary does not seem to affect the churn rate.


# Data Cleaning


### Removing Un-necessary columns

churn.head()

churn = churn.drop('Exited_str',axis=1)
churn

#### One-hot encoding our categorical data.


list_cat = ['Geography', 'Gender']
churn = pd.get_dummies(churn, columns = list_cat, prefix = list_cat)
churn.head()

churn.info()

### Splitting our data in to train and test dataset

from sklearn.model_selection import train_test_split , GridSearchCV

train, test = train_test_split(churn, test_size = 0.2, random_state= 1)
train.head(), test.head()

features = list(train.drop('Exited', axis = 1))
target = 'Exited'

### Getting the percentage of Exited data in both train test dataset

exited_train = len(train[train['Exited'] == 1]['Exited'])
exited_train_perc = round(exited_train/len(train)*100,1)

exited_test = len(test[test['Exited'] == 1]['Exited'])
exited_test_perc = round(exited_test/len(test)*100,1)

print('Complete Train set - Number of clients that have exited the program: {} ({}%)'.format(exited_train, exited_train_perc))
print('Test set - Number of clients that haven\'t exited the program: {} ({}%)'.format(exited_test, exited_test_perc))

#### Scaling our test and train data

NB_train, NB_test = train_test_split(churn, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train[features]= sc.fit_transform(train[features])
test[features] = sc.transform(test[features])

# Testing Different Models

####  Metrics functions import to test different predications from different algorithms

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### Decision Tree

from sklearn.tree import DecisionTreeClassifier

DT_Classify = DecisionTreeClassifier()
DT_Classify = DT_Classify.fit(train[features], train[target])

DT_pred = DT_Classify.predict(test[features])

test[target]

# Checking the accuracy of DecisionTreeClassifier

DT_Acc_Per= accuracy_score(test[target], DT_pred)
DT_Cla_Rep = classification_report(test[target], DT_pred)
print("Accuracy is",DT_Acc_Per)
print(DT_Cla_Rep)
cm = confusion_matrix(test[target], DT_pred)
plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Decision Tree", DT_Acc_Per)

### Multinomial Naive Bayes

The multinomial Naive Bayes classifier is suitable for classification with discrete features

from sklearn.naive_bayes import MultinomialNB


NB_Classifier = MultinomialNB()
NB_Classifier = NB_Classifier.fit(NB_train[features], NB_train[target])
NB_pred = NB_Classifier.predict(NB_test[features])

NB_Acc_Scr = accuracy_score(NB_test[target], NB_pred) 
NB_Cla_Rep = classification_report(NB_test[target], NB_pred)

cm = confusion_matrix(NB_test[target], NB_pred)
plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Multinomial Naive Bayes", NB_Acc_Scr)
print(NB_Cla_Rep)
"Accuracy is",NB_Acc_Scr

### KNN

from sklearn.neighbors import KNeighborsClassifier

KNN_Classifier = KNeighborsClassifier()
KNN_Classifier.fit(train[features], train[target])
KNN_pred= KNN_Classifier.predict(test[features])

KNN_acc = accuracy_score(test[target], KNN_pred)
KNN_Cla_Rep = classification_report(test[target], KNN_pred)

cm = confusion_matrix(test[target], KNN_pred)

plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"K- nearest neighbor", KNN_acc)

print(KNN_Cla_Rep)
"Accuraccy of KNN is",KNN_acc

### SVM - Supoort Vector Machines

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

from sklearn import svm

SVM_Classifier = svm.SVC(decision_function_shape='ovo')
SVM_Classifier = SVM_Classifier.fit(train[features], train[target])
SVM_pred = SVM_Classifier.predict(test[features])



cm = confusion_matrix(test[target], SVM_pred)


SVM_acc = accuracy_score(test[target], SVM_pred)

plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Support Vector Machine", SVM_acc)

SVM_Cla_Rep = classification_report(NB_test[target], SVM_pred)
print(SVM_Cla_Rep)

### Logistic Regression

Logistic Regression (aka logit, MaxEnt) classifier.

from sklearn.linear_model import LogisticRegression

parameters = {'C': [0.01, 0.1, 1, 10],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [50, 100, 150]}
LR = LogisticRegression(penalty = 'l2')
model_LR = GridSearchCV(LR, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_LR.cv_results_)


LR_Classifier = LogisticRegression(**model_LR.best_params_)

LR_Classifier = LR_Classifier.fit(train[features], train[target])
LR_pred = LR_Classifier.predict(test[features])


LR_acc = accuracy_score(test[target], LR_pred)
print("Logistic Regression accuracy is",LR_acc)
cm = confusion_matrix(test[target], LR_pred)
LR_Cla_Rep = classification_report(test[target], LR_pred)
print(LR_Cla_Rep)
plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Logistic Regression", LR_acc)


### SGD - Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.

from sklearn.linear_model import SGDClassifier

SGD_clf = SGDClassifier(loss='log', penalty="l1", max_iter=20, learning_rate="adaptive", eta0=0.01)
SGD_clf.fit(train[features], train[target])
SGD_pred = SGD_clf.predict(test[features])

SGD_acc = accuracy_score(test[target], SGD_pred)
print("SGD's accuracy is",SGD_acc)
cm = confusion_matrix(test[target], SGD_pred)
SGD_Cla_Rep = classification_report(test[target], SGD_pred)
print(SGD_Cla_Rep)
plot_Confusion_matrix(cm,['Exited','Not Exited'],'Blues',"Stochastic Gradient Descent", SGD_acc)



Show_Model_Score()


## Observations for Model Selection

1. Decision Tree - It performed well but the confusion graph shows that its has predicted some false Exited and False Not Exited but it can be considered due to satisfactory accuracy.

2. Multinomial Naive Bayes - This model doesn't even gave satisfactory results so we will not consider this.

3. KNN - This model has second most high accuracy and it is also very useful.

4. SVM - This model has the most promising performance as well as accuracy It will be very useful.

5. Logistic Regression - This model has very expectations as it is very efficient to predict multi class output and it performed well but it was not able to defeat SVM and it secured 4th place in consideration for model.

6. SGD (Stochastic Gradient Descent) - This was the most interesting and flexible algoritm it also performed well and it achieved 3rd most accurate model consideration place with some Hyperparameters tuning.

