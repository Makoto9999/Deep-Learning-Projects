import numpy as np
import pandas as pd
from pandas import DataFrame as df
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
seed=101

# Read CSV files and data sample
whiteread = pd.read_table('/path',
                          header=None,import pandas as pd
from pandas import DataFrame as df
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
seed=100

# Read CSV files and data sample
whiteread = pd.read_table('/path',
                          header=None,
                          sep=",",
                          names=["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","Y"])
print("Original file shape", whiteread.shape)
print(whiteread.head())

# Seperate Predictors and Target Data
X_numeric = whiteread[["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13"]]
Y_target = whiteread[["Y"]]

# Apply Z-socre to X_numeric
X_zsocre = df(X_numeric)
for col in X_zsocre:
    X_zsocre[col] = (X_zsocre[col] - X_zsocre[col].mean())/X_zsocre[col].std(ddof=0)
print(X_zsocre.head())
print(X_zsocre.shape)


# Change Y_target to binary data
Y_target = df(Y_target)
Y_target[Y_target["Y"] != 0] = 1
print(Y_target.head())
print(Y_target.shape)
print('Distribution of Y\n',Y_target['Y'].value_counts())

# Split Dataset 0.8:0.2
X_train,X_test,Y_train,Y_test=train_test_split(X_zsocre,Y_target,test_size = 0.2, random_state = seed)
print(X_train.head())
print(Y_train.head())
print(X_test.head())
print(Y_test.head())

# Build Neural Network
# Create Model
model=Sequential()
model.add(Dense(10,input_dim=13,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fit the model
model.fit(X_train,Y_train,epochs=100,batch_size=20)

# Summarize layers
model.summary()

# Evaluate the model
scores = model.evaluate(X_train,Y_train)
print("\n%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))
acc_train = model.metrics_names[1],scores[1]*100

# Fit the model on Test data
model.fit(X_test,Y_test,epochs=100,batch_size=20)

# Fit the data
Y_predict = model.predict(X_test)
Y_predict = Y_predict > 0.5

# Evaluate the model on Test data
scores = model.evaluate(X_test,Y_test)
print("\n%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))
acc_test = model.metrics_names[1],scores[1]*100

# Compare the Accuracy on both Training and Testing data
print("Acc for Train",acc_train)
print("Acc for Test",acc_test)
# print ('Precision:', metrics.precision_score(Y_test, Y_predict))
print ('Precision:', metrics.precision_score(Y_test, Y_predict))
print ('\n classification report:\n', metrics.classification_report(Y_test, Y_predict))
print ('\n confusion matrix:\n',metrics.confusion_matrix(Y_test, Y_predict))

# Plot ROC Curve - calculate fpr, tpr for class 0
Y_predict_class0 = 1- Y_predict
fpr_class0, tpr_class0, thresholds_class0 = metrics.roc_curve(1-Y_test,
Y_predict_class0)
auc_class0 = metrics.auc(fpr_class0, tpr_class0)
print("auc for class 0",auc_class0)

# Plot ROC Curve - calculate fpr, tpr for class 1
Y_predict_class1 = Y_predict
fpr_class1, tpr_class1, thresholds_class1 = metrics.roc_curve(Y_test,
Y_predict_class1)
auc_class1 = metrics.auc(fpr_class1, tpr_class1)
print("auc for class 1",auc_class1)

# Ploting the roc curve
plt.plot(fpr_class0,tpr_class0,color='red')
# plt.plot(fpr_class1,tpr_class1,color='green')
plt.plot(fpr_class0, tpr_class0, color='red',lw=2,
         label='ROC curve - class 0 (Not interesting) (area = %0.2f)' % auc_class0)
plt.plot(fpr_class1, tpr_class1, color='green',lw=2,
         label='ROC curve - class 1 (Interesting) (area = %0.2f)' % auc_class1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for payment default')
plt.xlabel('False positive rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

                          sep=",",
                          names=["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","Y"])
print("Original file shape", whiteread.shape)
print(whiteread.head())

# Seperate Predictors and Target Data
X_numeric = whiteread[["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13"]]
Y_target = whiteread[["Y"]]

# Apply Z-socre to X_numeric
X_zsocre = df(X_numeric)
for col in X_zsocre:
    X_zsocre[col] = (X_zsocre[col] - X_zsocre[col].mean())/X_zsocre[col].std(ddof=0)
print(X_zsocre.head())
print(X_zsocre.shape)


# Change Y_target to binary data
ohencod = preprocessing.OneHotEncoder()
Y_target = ohencod.fit(Y_target)
# Y_target = df(Y_target)
# Y_target[Y_target["Y"] != 0] = 1
# print(Y_target.head())
# print(Y_target.shape)
print('Distribution of Y\n',Y_target['Y'].value_counts())

# Split Dataset 0.8:0.2
X_train,X_test,Y_train,Y_test=train_test_split(X_zsocre,Y_target,test_size = 0.2, random_state = seed)
print(X_train.head())
print(Y_train.head())
print(X_test.head())
print(Y_test.head())

# Build Neural Network
# Create Model
model=Sequential()
model.add(Dense(10,input_dim=13,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fit the model
model.fit(X_train,Y_train,epochs=100,batch_size=20)

# Summarize layers
model.summary()

# Evaluate the model
scores = model.evaluate(X_train,Y_train)
print("\n%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))
acc_train = model.metrics_names[1],scores[1]*100

# Fit the model on Test data
model.fit(X_test,Y_test,epochs=100,batch_size=20)

# Fit the data
Y_score = model.predict_proba(X_test)
Y_predict = df(model.predict_proba(X_test))
Y_predict[Y_predict>0.5]=1
Y_predict[Y_predict<0.5]=0

# Evaluate the model on Test data
scores = model.evaluate(X_test,Y_test)
print("\n%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))
acc_test = model.metrics_names[1],scores[1]*100

# Compare the Accuracy on both Training and Testing data
print("Acc for Train",acc_train)
print("Acc for Test",acc_test)

# print ('Precision:', metrics.precision_score(Y_test, Y_predict))
print ('Precision:', metrics.precision_score(Y_test, Y_predict))
print ('\n classification report:\n', metrics.classification_report(Y_test, Y_predict))
print ('\n confusion matrix:\n',metrics.confusion_matrix(Y_test, Y_predict))

# Plot ROC Curve - calculate fpr, tpr for class 0
Y_predict_class0 = 1- Y_score
Y_predict_class0 = (np.array(Y_predict_class0)).flatten()
Y_test = np.array(Y_test).flatten()
fpr_class0, tpr_class0, thresholds_class0 = metrics.roc_curve(1-Y_test,Y_predict_class0)
auc_class0 = metrics.auc(fpr_class0, tpr_class0)
print("auc for class 0",auc_class0)

# Plot ROC Curve - calculate fpr, tpr for class 1
Y_predict_class1 = np.array(Y_score).flatten()
fpr_class1, tpr_class1, thresholds_class1 = metrics.roc_curve(Y_test,
Y_predict_class1)
auc_class1 = metrics.auc(fpr_class1, tpr_class1)
print("auc for class 1",auc_class1)

# Ploting the roc curve
plt.plot(fpr_class0,tpr_class0,color='red')
# plt.plot(fpr_class1,tpr_class1,color='green')
plt.plot(fpr_class0, tpr_class0, color='red',lw=2,
         label='ROC curve - class 0 (Not interesting) (area = %0.2f)' % auc_class0)
plt.plot(fpr_class1, tpr_class1, color='green',lw=2,
         label='ROC curve - class 1 (Interesting) (area = %0.2f)' % auc_class1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
