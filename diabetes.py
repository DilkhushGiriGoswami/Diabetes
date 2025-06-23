#import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix,classification_report
import joblib
from tensorflow import keras
#load the data set
df=pd.read_csv("diabetes.csv")

df.head()

#checking the basic information of dataset
df.info()
#divide the data set into the dependent and independent variables
x=df.drop(columns="Outcome")
y=df["Outcome"]
x.head()
scaler=StandardScaler()
x= scaler.fit_transform(x)
joblib.dump(scaler,"scaler.pkl")
#dividing the dataset into the Trainging and testing
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1)
X_train.shape
#Building a Neural Network
model=keras.Sequential([
    keras.layers.Dense(16,activation="relu",input_shape=(x.shape[1],)),#input layer
    keras.layers.Dense(8,activation="relu"),#Hidden layer
    keras.layers.Dense(1,activation="sigmoid")#Output Layer
])

#printing the summary of model
model.summary()
#compile the model
model.compile(optimizer="adam",loss="binary_crossentropy")
#train the model
model.fit(X_train,Y_train,epochs=50,batch_size=10,validation_data=(X_test,Y_test))
y_predict=model.predict(X_test)
y_predict=(y_predict>0.5).astype('int32')
#calculate the performance matrix
accuracy=accuracy_score(y_predict,Y_test)
recall=recall_score(y_predict,Y_test)
precision=precision_score(y_predict,Y_test)
f1=f1_score(y_predict,Y_test)
cm=confusion_matrix(y_predict,Y_test)
cr=classification_report(y_predict,Y_test)
print("The accuracy of our Diabetes prediction Model is : ",accuracy)
print("The recall of our Diabetes prediction Model is : ",recall)
print("The precision of our Diabetes prediction Model is : ",precision)
print("The f1 score of our Diabetes prediction Model is : ",f1)
print("The confusion matrix of our Diabetes prediction Model is : ",cm)
print("The classification report of our Diabetes prediction Model is : ",cr)
model.save("Diabetic_model.h5")
print("Saved model has been saved")