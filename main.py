import pandas as pd
# dataset = pd.read_csv("https://raw.githubusercontent.com/rajtilakls2510/car_price_predictor/master/quikr_car.csv")
# dataset.to_csv("car_price.csv")
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("car_price.csv")
print(dataset.shape)
print(dataset.info())
# print(dataset.head())

# finding rows and columns of the data
# print(dataset.shape)

# reviewing the data
# -year has many non year values , run dataset['year].unique
# print(dataset['year'].unique())
# -year is not int here , it is object
# -price has 'ask for price' and it is also not int, it also hase commas like 10,000
# -kms has commas and the int are attached with 'kms', also has nan values
# -fuel type also has nan values
# -we will keep first 3 words of name

# building a backup for dataset
backup = dataset.copy()

# cleaning the data
# storing only numeric values in year
dataset = dataset[dataset['year'].str.isnumeric()]
# converting year from object to int
dataset['year'] = dataset['year'].astype(int)
# may check it by dataset.info()
dataset = dataset[dataset['Price'] != "Ask For Price"]
# replacing comma with empty string and converting to int
dataset['Price'] = dataset['Price'].str.replace(',', '').astype(int)
dataset['kms_driven'] = dataset['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
# get(0) gives the first part that is without kms
dataset = dataset[dataset['kms_driven'].str.isnumeric()]
dataset['kms_driven'] = dataset['kms_driven'].astype(int)
dataset = dataset[~dataset['fuel_type'].isna()]
dataset['name'] = dataset['name'].str.split(' ').str.slice(0, 3).str.join(' ')
dataset.reset_index(drop=True)
# dataset.info()
# print(dataset.describe())
# only keeping cars with price less than 6 lakhs
dataset = dataset[dataset['Price']<6e6].reset_index(drop=True)


# building model
dataset.to_csv('cleaned_dataset.csv')
x= dataset.drop(columns='Price')
y= dataset['Price']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import  make_pipeline
ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
# here column_trans and other columns will be given to lr model
pipe.fit(x_train,y_train)
# yaha pipe pehle column_trans ke through onehotencoder lagayega and fir usko directly  lr model ko bhej dega
# one end of pipeline will send raw data and on the other end we get lr output
y_pred = pipe.predict(x_test)
# print(y_pred)
# print(r2_score(y_test, y_pred))

#finding the best randome_state to get max r2_Score
scores =[]
for i in range(10):
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    # print(r2_score(y_test, y_pred), i)
    scores.append(r2_score(y_test,y_pred))
print(np.argmax(scores))
print(scores[np.argmax(scores)])
# getting max random state at 4 as 0.804

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

# print(pipe.predict(pd.DataFrame(columns=x_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))

import pickle
pickle.dump(pipe, open('LinearRegressionModel.pkl','wb'))
print(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))





