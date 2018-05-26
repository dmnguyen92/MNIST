import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Display image
def displayImage(X,setting):
    img = np.resize(X,(28,28))
    plt.imshow(img,cmap=setting)
    plt.axis('off')

# Import data
trainSet = pd.read_csv('Data/train.csv')
X = trainSet.iloc[:,1:785].values
y = trainSet.iloc[:,0].values

# Encode data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
# onehotencoder_y = OneHotEncoder(categorical_features=[0])
y = labelencoder_y.fit_transform(y)
    
# Train_test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Apply SVM
from sklearn.svm import SVC
clf = SVC(kernel = 'poly', degree = 2, random_state = 0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

# Predict final result
testSet = pd.read_csv('Data/test.csv')
clf = SVC(kernel = 'poly', degree = 2, random_state = 0)
clf.fit(X,y)
label = clf.predict(testSet)

# Output to csv file
filename = 'result.csv'
myfile = open(filename,'w')
titleRow = 'ImageId, Label\n'
myfile.write(titleRow)
index = 0
for y in label:
    index = index+1
    row = str(index) + ',' + str(y) + '\n'
    myfile.write(row)
        




