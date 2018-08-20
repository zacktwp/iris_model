
# coding: utf-8

# In[1]:


# Multiclass Classification with the Iris Flowers Dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
#import boto3
#import s3fs
#import os
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


prefix = '/opt/ml/'
#model_path = os.path.join(prefix, 'model')


#load data from s3
#bucket_name = 'iris-docker-data'
#s3 = boto3.client('s3')
#obj = s3.get_object(Bucket=bucket_name, Key='iris.csv')
dataframe = pd.read_csv('iris.csv')


# In[3]:


dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]


# In[4]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[10]:


# define baseline model
def iris_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, dummy_y, epochs=200, batch_size=5, verbose=0, shuffle=True)
    model.save('iris_model.h5')
    return model


# In[11]:


iris_model()


# In[34]:


from keras.models import load_model
model = load_model('iris_model.h5')
print(moddel.predict(X))

