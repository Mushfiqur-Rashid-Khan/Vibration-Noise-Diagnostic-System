import pandas as pd 
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

data = pd.read_csv('Dataset.csv') 
print (data.head())

y = data.Pred  
X = data.drop('Pred', axis=1) 
train, test, train_label, test_label = train_test_split(X,y,test_size=0.2) 

from keras.models import Sequential 
from keras.layers import Dense 
 
# Call CNN model 
model = Sequential() 
model.add(Dense(12, input_dim=20, activation='relu')) 
model.add(Dense(8, activation='relu')) 
#model.add(Dense(1, activation='sigmoid')) 

# Compile model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Fit the model 
history=model.fit(train, train_label, epochs=20, batch_size=32) 
# evaluate the model 
scores = model.evaluate(test, test_label) 
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 
