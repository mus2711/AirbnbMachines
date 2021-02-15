import pandas as pd 
import warnings
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import graphviz
pd.set_option('display.max_rows', 500)

def select_column_to_add(X_train, y_train, X_val, y_val, columns_in_model, columns_to_test):
    
    column_best = None
    columns_in_model = list(columns_in_model)
    
    if len(columns_in_model) == 0:
        acc_best = 0
    elif len(columns_in_model) == 1:
        mod = LogisticRegression(C=1e9).fit(X_train[columns_in_model].values.reshape(-1, 1), y_train)
        acc_best = accuracy_score(y_val, mod.predict(X_val[columns_in_model].values.reshape(-1, 1)))
    else:
        mod = LogisticRegression(C=1e9).fit(X_train[columns_in_model], y_train)
        acc_best = accuracy_score(y_val, mod.predict(X_val[columns_in_model]))

    
    for column in columns_to_test:
        mod = LogisticRegression(C=1e9).fit(X_train[columns_in_model+[column]], y_train)
        y_pred = mod.predict(X_val[columns_in_model+[column]])
        acc = accuracy_score(y_val, y_pred)
        
        if acc - acc_best >= 0.005:  # one of our stopping criteria
            acc_best = acc
            column_best = column
        
    if column_best is not None:  # the other stopping criteria
        print('Adding {} to the model'.format(column_best))
        print('The new best validation accuracy is {}'.format(acc_best))
        columns_in_model_updated = columns_in_model + [column_best]
    else:
        print('Did not add anything to the model')
        columns_in_model_updated = columns_in_model
    
    return columns_in_model_updated, acc_best

class ModelSummary:
    """ This class extracts a summary of the model
    
    Methods
    -------
    get_se()
        computes standard error
    get_ci(SE_est)
        computes confidence intervals
    get_pvals()
        computes p-values
    get_summary(name=None)
        prints the summary of the model
    """
    
    def __init__(self, clf, X, y):
        """
        Parameters
        ----------
        clf: class
            the classifier object model
        X: pandas Dataframe
            matrix of predictors
        y: numpy array
            matrix of variable
        """
        self.clf = clf
        self.X = X
        self.y = y
        pass
    
    def get_se(self):
        """Computes the standard error

        Returns
        -------
            numpy array of standard errors
        """
        # from here https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
        predProbs = self.clf.predict_proba(self.X)
        X_design = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
        V = np.diagflat(np.product(predProbs, axis=1))
        covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
        return np.sqrt(np.diag(covLogit))

    def get_ci(self, SE_est):
        """Computes the confidence interval

        Parameters
        ----------
        SE_est: numpy array
            matrix of standard error estimations
        
        Returns
        -------
        cis: numpy array
            matrix of confidence intervals
        """
        p = 0.975
        df = len(self.X) - 2
        crit_t_value = stats.t.ppf(p, df)
        coefs = np.concatenate([self.clf.intercept_, self.clf.coef_[0]])
        upper = coefs + (crit_t_value * SE_est)
        lower = coefs - (crit_t_value * SE_est)
        cis = np.zeros((len(coefs), 2))
        cis[:,0] = lower
        cis[:,1] = upper
        return cis
    
    def get_pvals(self):
        """Computes the p-value

        Returns
        -------
        p: numpy array
            matrix of p-values
        """
        # from here https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
        p = self.clf.predict_proba(self.X)
        n = len(p)
        m = len(self.clf.coef_[0]) + 1
        coefs = np.concatenate([self.clf.intercept_, self.clf.coef_[0]])
        se = self.get_se()
        t =  coefs/se  
        p = (1 - stats.norm.cdf(abs(t))) * 2
        return p
    
    def get_summary(self, names=None):
        """Prints the summary of the model

        Parameters
        ----------
        names: list
            list of the names of predictors
        """
        ses = self.get_se()
        cis = self.get_ci(ses)
        lower = cis[:, 0]
        upper = cis[:, 1]
        pvals = self.get_pvals()
        coefs = np.concatenate([self.clf.intercept_, self.clf.coef_[0]])
        data = []
        for i in range(len(coefs)):
            currlist = []
            currlist.append(np.round(coefs[i], 3))
            currlist.append(np.round(ses[i], 3))
            currlist.append(np.round(pvals[i], 3))
            currlist.append(np.round(lower[i], 3))
            currlist.append(np.round(upper[i], 3))
            data.append(currlist)
        cols = ['coefficient', 'std', 'p-value', '[0.025', '0.975]']
        sumdf = pd.DataFrame(columns=cols, data=data)
        if names is not None:
            new_names = ['intercept']*(len(names) + 1)
            new_names[1:] = [i for i in names]
            sumdf.index = new_names
        else:
            try:
                names = list(self.X.columns)
                new_names = ['intercept']*(len(names) + 1)
                new_names[1:] = [i for i in names]
                sumdf.index = new_names
            except:
                pass
        print(sumdf)
        acc = accuracy_score(self.y, self.clf.predict(self.X))
        confmat = confusion_matrix(self.y, self.clf.predict(self.X))
        print('-'*60)
        print('Confusion Matrix (total:{}) \t Accuracy: \t  {}'.format(len(self.X),np.round(acc, 3)))
        print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
        print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))


warnings.filterwarnings("ignore")

X_data = pd.read_csv('x_cleaned.csv')
y_data = pd.read_csv('y_cleaned.csv')

y_datacat = pd.cut(y_data.cleaned_price,bins=[0, 100.0, 1000.0],labels=['0', '1'])

X_train, otherX = train_test_split(X_data, test_size=0.2, random_state=0)
X_test, X_val = train_test_split(otherX, test_size=0.5, random_state=0)

y_train, otherY = train_test_split(y_datacat, test_size=0.2, random_state=0)
y_test, y_val = train_test_split(otherY, test_size=0.5, random_state=0)

columns = ['room_type_Entire home_apt', 'bedrooms', 'neighbourhood_group_cleansed_Ballard', 
'neighbourhood_group_cleansed_Beacon_Hill', 'neighbourhood_group_cleansed_Delridge']

traincol = []
for col in X_train.columns:
    traincol.append(col)

for col in columns:
     traincol.remove(col)

  
# select_column_to_add(X_train, y_train, X_val, y_val, columns, traincol)
print('-'*60)
print('LineaerRegression Model, Summary on Validation Set')
model = LogisticRegression()
model.fit(X_train[columns],y_train)

modsummary = ModelSummary(model, X_val[columns], y_val)
modsummary.get_summary()

ypredtr = model.predict(X_train[columns])
acctr = accuracy_score(y_train, ypredtr) 
prectr = precision_score(y_train, ypredtr, pos_label='1') 
rectr = recall_score(y_train, ypredtr, pos_label='1')

ypred = model.predict(X_val[columns])
acc = accuracy_score(y_val, ypred) 
prec = precision_score(y_val, ypred, pos_label='1') 
rec = recall_score(y_val, ypred, pos_label='1')

ypredt = model.predict(X_test[columns])
acct = accuracy_score(y_test, ypredt) 
prect = precision_score(y_test, ypredt, pos_label='1') 
rect = recall_score(y_test, ypredt, pos_label='1')

print('Train Precision:{}'.format(prectr)) 
print('Train Recall: {}'.format(rectr)) 
print('Train Accuracy: {}'.format(acctr))

print('Validaiton Precision:{}'.format(prec)) 
print('Validation Recall: {}'.format(rec)) 
print('Validation Accuracy: {}'.format(acc))

print('Test Precision:{}'.format(prect)) 
print('Test Recall: {}'.format(rect)) 
print('Test Accuracy: {}'.format(acct))

confmat = confusion_matrix(y_train, model.predict(X_train[columns]))
print('Confusion Matrix, Train Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

confmat = confusion_matrix(y_val, model.predict(X_val[columns]))
print('Confusion Matrix, Validation Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

confmat = confusion_matrix(y_test, model.predict(X_test[columns]))
print('Confusion Matrix, Test Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))





print('-'*60)
print('Tree, Max Depth of 8')
clf1 = tree.DecisionTreeClassifier(max_depth=8) 
clf1.fit(X_train, y_train)
print('Accuracy on the train set: {}'.format(accuracy_score(y_train, clf1.predict(X_train))))
print('Accuracy on the val set: {}'.format(accuracy_score(y_val, clf1.predict(X_val))))
print('Accuracy on the test set: {}'.format(accuracy_score(y_test, clf1.predict(X_test))))
print('Precision on the train set: {}'.format(precision_score(y_train, clf1.predict(X_train), pos_label='1')))
print('Precision on the val set: {}'.format(precision_score(y_val, clf1.predict(X_val), pos_label='1')))
print('Precision on the test set: {}'.format(precision_score(y_test, clf1.predict(X_test), pos_label='1')))
print('Recall on the train set: {}'.format(recall_score(y_train, clf1.predict(X_train), pos_label='1')))
print('Recall on the val set: {}'.format(recall_score(y_val, clf1.predict(X_val), pos_label='1')))
print('Recall on the test set: {}'.format(recall_score(y_test, clf1.predict(X_test), pos_label='1')))

confmat = confusion_matrix(y_val, clf1.predict(X_val))
print('Confusion Matrix, Validation Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))
confmat = confusion_matrix(y_train, clf1.predict(X_train))
print('Confusion Matrix, Train Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))
confmat = confusion_matrix(y_test, clf1.predict(X_test))
print('Confusion Matrix, Test Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

max_depth_list = list(range(1,26))
train_acc = [] 
val_acc = []
train_pre = []
val_pre = []
train_rec = []
val_rec = []
for x in max_depth_list:
    dtc = tree.DecisionTreeClassifier(max_depth=x, min_impurity_decrease=0.01595) 
    dtc.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    val_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = np.arange(len(max_depth_list)) + 1 # Create domain for plot
plt.xticks(np.arange(1, 26, step=1))
plt.plot(x, train_acc, label='Training Accuracy') 
plt.plot(x, val_acc, label='Validation Accuracy')
plt.xlabel('Maximum Depth') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

for x in max_depth_list:
    dtc = tree.DecisionTreeClassifier(max_depth=x,  min_impurity_decrease=0.01595) 
    dtc.fit(X_train, y_train)
    train_pre.append(precision_score(y_train, dtc.predict(X_train), pos_label='1'))
    val_pre.append(precision_score(y_val, dtc.predict(X_val), pos_label='1'))
 
x = np.arange(len(max_depth_list)) + 1 # Create domain for plot
plt.xticks(np.arange(1, 26, step=1))
plt.plot(x, train_pre, label='Training Precision')
plt.plot(x, val_pre, label='Validation Precision') 
plt.xlabel('Maximum Depth') # Label x-axis
plt.ylabel('Precision') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

for x in max_depth_list:
    dtc = tree.DecisionTreeClassifier(max_depth=x,  min_impurity_decrease=0.01595) 
    dtc.fit(X_train, y_train)
    train_rec.append(recall_score(y_train, dtc.predict(X_train), pos_label='1'))
    val_rec.append(recall_score(y_val, dtc.predict(X_val), pos_label='1'))
 
x = np.arange(len(max_depth_list)) + 1 # Create domain for plot
plt.xticks(np.arange(1, 26, step=1))
plt.plot(x, train_rec, label='Training Recall')
plt.plot(x, val_rec, label='Validation Recall') 
plt.xlabel('Maximum Depth') # Label x-axis
plt.ylabel('Recall') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

min_imp_list = np.arange(0.015, 0.016, step=0.0001).tolist()
prtrain_acc = [] 
prval_acc = []
prtrain_pre = []
prval_pre = []
prtrain_rec = []
prval_rec = []

for x in min_imp_list:
    dtc = tree.DecisionTreeClassifier(max_depth=8, min_impurity_decrease=x) 
    dtc.fit(X_train, y_train)
    prtrain_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    prval_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = min_imp_list
plt.xticks(np.arange(0.015, 0.016, step=0.0001))
plt.plot(x, prtrain_acc, label='Training Accuracy') 
plt.plot(x, prval_acc, label='Validation Accuracy') 
plt.xlabel('Min Impurity Decrease') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

for x in min_imp_list:
    dtc = tree.DecisionTreeClassifier(max_depth=8, min_impurity_decrease=x) 
    dtc.fit(X_train, y_train)
    prtrain_pre.append(precision_score(y_train, dtc.predict(X_train), pos_label='1'))
    prval_pre.append(precision_score(y_val, dtc.predict(X_val), pos_label='1'))
 
x = min_imp_list
plt.xticks(np.arange(0.015, 0.016, step=0.0001))
plt.plot(x, prtrain_pre, label='Training Precision') 
plt.plot(x, prval_pre, label='Validation Precision') 
plt.xlabel('Min Impurity Decrease') # Label x-axis
plt.ylabel('Precision') # Label y-axis
print(max(y))
plt.legend() # Show plot labels as legend
plt.show() # Show graph

for x in min_imp_list:
    dtc = tree.DecisionTreeClassifier(max_depth=8, min_impurity_decrease=x) 
    dtc.fit(X_train, y_train)
    prtrain_rec.append(recall_score(y_train, dtc.predict(X_train), pos_label='1'))
    prval_rec.append(recall_score(y_val, dtc.predict(X_val), pos_label='1'))

x = min_imp_list 
plt.xticks(np.arange(0.015, 0.016, step=0.0001))
plt.plot(x, prtrain_rec, label='Training Recall') 
plt.plot(x, prval_rec, label='Validation Recall') 
plt.xlabel('Min Impurity Decrease') # Label x-axis
plt.ylabel('Recall') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

predictors = X_train.columns
dot_data = tree.export_graphviz(clf1, out_file='tree3.dot',
                                feature_names = predictors,
                                class_names = ('Below', 'Above'),
                                filled = True, rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)  


print('-'*60)
print('RFC, Max Depth of 14 and Minimum Sample Split of 500')
model = RandomForestClassifier(max_depth=14, min_samples_split=500)
model = model.fit(X_train, y_train) 

ypredtr = model.predict(X_train)
acctr = accuracy_score(y_train, ypredtr) 
prectr = precision_score(y_train, ypredtr, pos_label='1') 
rectr = recall_score(y_train, ypredtr, pos_label='1')

ypred = model.predict(X_val)
acc = accuracy_score(y_val, ypred) 
prec = precision_score(y_val, ypred, pos_label='1') 
rec = recall_score(y_val, ypred, pos_label='1')

ypredt = model.predict(X_test)
acct = accuracy_score(y_test, ypredt) 
prect = precision_score(y_test, ypredt, pos_label='1') 
rect = recall_score(y_test, ypredt, pos_label='1')

print('Train Precision:{}'.format(prectr)) 
print('Train Recall: {}'.format(rectr)) 
print('Train Accuracy: {}'.format(acctr))

print('Validaiton Precision:{}'.format(prec)) 
print('Validation Recall: {}'.format(rec)) 
print('Validation Accuracy: {}'.format(acc))

print('Test Precision:{}'.format(prect)) 
print('Test Recall: {}'.format(rect)) 
print('Test Accuracy: {}'.format(acct))

confmat = confusion_matrix(y_train, model.predict(X_train))
print('Confusion Matrix, Train Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

confmat = confusion_matrix(y_val, model.predict(X_val))
print('Confusion Matrix, Validation Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

confmat = confusion_matrix(y_test, model.predict(X_test))
print('Confusion Matrix, Test Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

estimator = model.estimators_[5]
dot_data = tree.export_graphviz(estimator, out_file='foresttree.dot',
                                feature_names = predictors,
                                class_names = ('Below', 'Above'),
                                filled = True, rounded = True,
                                special_characters = True)


max_depth_list = list(range(1,26))
train_acc = [] 
val_acc = []
train_pre = []
val_pre = []
train_rec = []
val_rec = []
for x in max_depth_list:
    dtc = RandomForestClassifier(max_depth=x) 
    dtc.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    val_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = np.arange(len(max_depth_list)) + 1 # Create domain for plot
plt.xticks(np.arange(1, 26, step=1))
plt.plot(x, train_acc, label='Training Accuracy') 
plt.plot(x, val_acc, label='Validation Accuracy')
plt.xlabel('Maximum Depth') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

max_n_list = np.arange(10, 100, step=10).tolist()
train_acc = [] 
val_acc = []
train_pre = []
val_pre = []
train_rec = []
val_rec = []
for x in max_n_list:
    dtc = RandomForestClassifier(max_depth=14, n_estimators=x) 
    dtc.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    val_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = max_n_list # Create domain for plot
plt.xticks(np.arange(10, 100, step=10))
plt.plot(x, train_acc, label='Training Accuracy') 
plt.plot(x, val_acc, label='Validation Accuracy') 
plt.xlabel('n_estimator') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph



print('-'*60)
print('SVC, C Value of 1.7, Using rbf kernal and a gamma setting scale')
model = SVC(C=1.7, kernel='rbf', gamma='scale')
model.fit(X_train, y_train)

ypredtr = model.predict(X_train)
acctr = accuracy_score(y_train, ypredtr) 
prectr = precision_score(y_train, ypredtr, pos_label='1') 
rectr = recall_score(y_train, ypredtr, pos_label='1')

ypred = model.predict(X_val)
acc = accuracy_score(y_val, ypred) 
prec = precision_score(y_val, ypred, pos_label='1') 
rec = recall_score(y_val, ypred, pos_label='1')


ypredt = model.predict(X_test)
acct = accuracy_score(y_test, ypredt) 
prect = precision_score(y_test, ypredt, pos_label='1') 
rect = recall_score(y_test, ypredt, pos_label='1')

print('Train Precision:{}'.format(prectr)) 
print('Train Recall: {}'.format(rectr)) 
print('Train Accuracy: {}'.format(acctr))

print('Validaiton Precision:{}'.format(prec)) 
print('Validation Recall: {}'.format(rec)) 
print('Validation Accuracy: {}'.format(acc))

print('Test Precision:{}'.format(prect)) 
print('Test Recall: {}'.format(rect)) 
print('Test Accuracy: {}'.format(acct))

confmat = confusion_matrix(y_train, model.predict(X_train))
print('Confusion Matrix, Train Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

confmat = confusion_matrix(y_val, model.predict(X_val))
print('Confusion Matrix, Validation Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

confmat = confusion_matrix(y_test, model.predict(X_test))
print('Confusion Matrix, Test Set')
print('  TP: {} | FN: {}'.format(confmat[1][1],confmat[1][0]))
print('  FP: {} | TN: {}'.format(confmat[0][1],confmat[0][0]))

c_list = np.arange(0.1, 5, step=0.2).tolist()
train_acc = [] 
val_acc = []
train_pre = []
val_pre = []
train_rec = []
val_rec = []
for x in c_list:
    dtc = SVC(C=x) 
    dtc.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    val_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = c_list # Create domain for plot
plt.xticks(np.arange(0.1, 5, step=0.2))
plt.plot(x, train_acc, label='Training Accuracy') 
plt.plot(x, val_acc, label='Validation Accuracy') 
plt.xlabel('C Value') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

kernal_list = ['linear', 'poly', 'rbf', 'sigmoid']
train_acc = [] 
val_acc = []
train_pre = []
val_pre = []
train_rec = []
val_rec = []
for x in kernal_list:
    dtc = SVC(C=1.7, kernel=x) 
    dtc.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    val_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = kernal_list # Create domain for plot
plt.plot(x, train_acc, label='Training Accuracy') 
plt.plot(x, val_acc, label='Validation Accuracy') 
plt.xlabel('Kernal') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

gamma_list = ['auto', 'scale', 0.005]
train_acc = [] 
val_acc = []
train_pre = []
val_pre = []
train_rec = []
val_rec = []
for x in gamma_list:
    dtc = SVC(C=1.7, kernel='rbf', gamma=x) 
    dtc.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, dtc.predict(X_train)))
    val_acc.append(accuracy_score(y_val, dtc.predict(X_val)))
 
x = gamma_list # Create domain for plot
plt.plot(x, train_acc, label='Training Accuracy')
plt.plot(x, val_acc, label='Validation Accuracy') 
plt.xlabel('Gamma') # Label x-axis
plt.ylabel('Accuracy') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph