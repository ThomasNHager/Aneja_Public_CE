#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Install lifelines if needed
#!pip install lifelines


# In[146]:


#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import pairwise_logrank_test
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers import Activation
from tensorflow import keras


# In[5]:


#Reading the data in
data = pd.read_csv("Aneja_Coding_Exercise_KM_updated.csv")
data


# In[6]:


#Diving the data based on group
test = data[data['Group'] == 'miR-137']
control = data[data['Group'] == 'control']
display(test)
display(control)


# In[7]:


#Kaplan Meier Curve
kmf = KaplanMeierFitter()
kmf.fit(test['Time'], test['Event'], label = 'miR-137')
plt.figure(figsize=(8,4))
kmf.plot()
kmf.fit(control['Time'], control['Event'], label = 'Control')
kmf.plot()
plt.title("Kaplan Meier Estimate By Group")
plt.xlabel("Follow Up Time")
plt.ylabel("Survival")
#Saving the first plot
plt.savefig("Aneja KM Curves.png")


# In[8]:


#Running the logrank test
logrank = pairwise_logrank_test(data['Time'], data['Group'], data['Event'])
logrank.summary


# In[9]:


#Separating the original data based on quartile
Quantiles = data.Predicted.quantile([0.25, 0.5, 0.75])
QData = data

conditions = [
    (QData['Predicted'] <= Quantiles[0.25]),
    ((QData['Predicted'] > Quantiles[0.25]) & (QData['Predicted'] <= Quantiles[0.5])),
    ((QData['Predicted'] > Quantiles[0.5]) & (QData['Predicted'] <= Quantiles[0.75])),
    (QData['Predicted'] > Quantiles[0.75])
]

values = [0, 1, 2, 3]

QData['Qt'] = np.select(conditions, values)
QData


# In[10]:


Q0 = QData[QData['Qt'] == 0]
Q1 = QData[QData['Qt'] == 1]
Q2 = QData[QData['Qt'] == 2]
Q3 = QData[QData['Qt'] == 3]


# In[11]:


#Kaplan Meier Curve Based on Quantiles
kmf = KaplanMeierFitter()
kmf.fit(Q0['Time'], Q0['Event'], label = 'First Quartile')
plt.figure(figsize=(8,4))
kmf.plot()
kmf.fit(Q1['Time'], Q1['Event'], label = 'Second Quartile')
kmf.plot()
kmf.fit(Q2['Time'], Q2['Event'], label = 'Third Quartile')
kmf.plot()
kmf.fit(Q3['Time'], Q3['Event'], label = 'Fourth Quartile')
kmf.plot()
plt.title("Kaplan Meier Estimate By Predicted Quartile")
plt.xlabel("Follow Up Time")
plt.ylabel("Survival")
#Saving the second plot
plt.savefig("Aneja KM Curves 2.png")


# In[12]:


#Running the logrank test for the quartile data
logrank = pairwise_logrank_test(QData['Time'], QData['Qt'], QData['Event'])
logrank.summary


# In[13]:


#Import the mnist dataset and break into train and test
mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
(xtrain, ytrain), (xtest, ytest) = mnist


# In[164]:


#Building the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (2,2), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(20))
model.summary()


# In[165]:


#Compiling and training the model
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

res = model.fit(x=xtrain, y=ytrain, epochs = 5, validation_data = (xtest, ytest))


# In[166]:


#Graphing test and validation loss
plt.plot(res.history['loss'], label = 'Test Loss')
plt.plot(res.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.title('Loss By Epoch')
plt.savefig("Aneja TV.png")


# In[167]:


#Accuracy
test_loss, test_acc = model.evaluate(xtest,  ytest, verbose=2)
print(test_acc)


# In[168]:


#AUC
auc = tf.keras.metrics.AUC(from_logits = True)
auc.update_state(ytest, model.predict(xtest)[:,1])
print(auc.result().numpy())


# In[ ]:




