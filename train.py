import numpy as np
import pandas as pd
df = pd.read_excel("CT -new data.xlsx",sheet_name="Sheet3")
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#print(df)
df = df.replace(r'\u200b', '', regex=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna() 
X = df.drop(['target'], axis = 1)
#print(X)
y = df['target']
#print(y)
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 5)
model = GaussianNB().fit(X_train, y_train)
prediction_test = model.predict(X_test)
'''print(accuracy_score(prediction_test,y_test))
print(prediction_test)
print(y_test)'''

import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
# Make predictions using the loaded mode