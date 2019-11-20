import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np


#load the database
db = pd.read_json('train_dataset.jsonl', lines=True)

#Define X and y from the database
X_all = db.loc[:, 'instructions']
y1_all = db.loc[:, 'compiler']
y2_all = db.loc[:, 'opt']

#choice to consider just the mnemonic or the entire instruction
'''
for x in X_all:
    y=0
    while(y<len(x)):
        temp = x[y].split(' ')[0]
        x[y] = temp
        y=y+1
'''
i = 0
lung = len(X_all)

while(i < lung):
    X_all[i] = ' '.join(X_all[i])
    i = i+1

#control print
print("Data: ")
print(X_all)


#choices of vectorizer
#vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,4))

X_all = vectorizer.fit_transform(X_all)

#COMPILER PART:
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_all, y1_all, test_size=0.25, random_state=14)

print("Compiler provenance")
#model1 = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 1000000)
model1 = svm.LinearSVC(fit_intercept=False, C=10)
#model1 = svm.LinearSVC()
#model1 = svm.SVC()
model1.fit(X_train1, y_train1)
y_pred1=model1.predict(X_test1)

acc1 = model1.score(X_test1, y_test1)
print("Accuracy %.3f" %acc1)
print(classification_report(y_test1, y_pred1))
print("Confusion matrix")
cm1 = confusion_matrix(y_test1, y_pred1, labels=None, sample_weight=None)
print(cm1)

#Tuning hyper-parameters
'''
parameters = {'kernel':['linear', 'poly'], 'C':[0.1, 1, 10]  }
modelclass = svm.SVC(gamma='scale') 
gridmodel = GridSearchCV(modelclass, parameters, cv=5, iid=False, n_jobs=-1)
gridmodel.fit(X_all, y1_all)


#print(gridmodel.cv_results_)

for i in range(0,len(gridmodel.cv_results_['params'])):
    print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
        gridmodel.cv_results_['params'][i],
        gridmodel.cv_results_['mean_test_score'][i],
        gridmodel.cv_results_['std_test_score'][i] ))

a = np.argmax(gridmodel.cv_results_['mean_test_score'])
bestparams = gridmodel.cv_results_['params'][a]
bestscore = gridmodel.cv_results_['mean_test_score'][a]

print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
print("Best kernel: %s" %(bestparams['kernel']))
print("Best C: %s" %(bestparams['C']))
'''

#OPTIMIZATION PART:
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all, y2_all, test_size=0.25, random_state=14)

print("Optimization")
#model2 = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 1000000)
model2 = svm.LinearSVC(fit_intercept=False, C=10)
#model2 = svm.LinearSVC()
model2.fit(X_train2, y_train2)
y_pred2 = model2.predict(X_test2)

acc2 = model2.score(X_test2, y_test2)
print("Accuracy %.3f" %acc2)
print(classification_report(y_test2, y_pred2))
print("Confusion matrix")
cm2 = confusion_matrix(y_test2, y_pred2, labels=None, sample_weight=None)
print(cm2)


#Tuning hyper-parameters
'''
parameters = {'kernel':['linear', 'poly'], 'C':[0.1, 1, 10]  }
modelclass = svm.SVC(gamma='scale') 
gridmodel = GridSearchCV(modelclass, parameters, cv=5, iid=False, n_jobs=-1)
gridmodel.fit(X_all, y1_all)


#print(gridmodel.cv_results_)

for i in range(0,len(gridmodel.cv_results_['params'])):
    print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
        gridmodel.cv_results_['params'][i],
        gridmodel.cv_results_['mean_test_score'][i],
        gridmodel.cv_results_['std_test_score'][i] ))

a = np.argmax(gridmodel.cv_results_['mean_test_score'])
bestparams = gridmodel.cv_results_['params'][a]
bestscore = gridmodel.cv_results_['mean_test_score'][a]

print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
print("Best kernel: %s" %(bestparams['kernel']))
print("Best C: %s" %(bestparams['C']))
'''

#BLIND DATASET TEST: the best configuration found was used for predictions on the blind dataset
#load blind dataset
dblind = pd.read_json(open("/home/george/Desktop/test_dataset_blind.jsonl", "r", encoding="utf8"),lines=True)
X_blind = dblind.loc[:, 'instructions']

i = 0
lung = len(X_blind)
while(i < lung):
    X_blind[i] = ' '.join(X_blind[i])
    i = i+1

#vectorizer
X_blind = vectorizer.transform(X_blind)

#compiler
y_pred_compiler = model1.predict(X_blind)

#optimization
y_pred_optimization = model2.predict(X_blind)

#print the prediction on a csv file in output
pd.DataFrame( data = {'Compiler' : y_pred_compiler, 'Optimization' : y_pred_optimization}).to_csv('prediction.csv', index = False)
