#First import libraries
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

#import the json file
f = '../input/iris-assignment/vertopal.com_algoparams_from_ui.json.json'
with open(f, 'r', encoding='utf-8') as f:
    my_data = json.load(f)

#import the dataset
data = pd.read_csv('../input/iris-assignment/iris.csv')
#print("check if there are some missimng values")
#print(data.isnull().sum())
# The data has no misssing value, so we don't need to impute any values 

for element in my_data['design_state_data']["algorithms"]:
    if my_data['design_state_data']["algorithms"][element]['is_selected'] == True:
        #print(element)
        print(" ")


#in the json file it is said that only random forest regressor is selected. 
#So we will work with random forest regressor

le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
y = data['petal_width']
data.drop('petal_width', axis = 1, inplace = True)


X = data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

rf = RandomForestRegressor(max_depth = 25 )
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#print("running clear")
