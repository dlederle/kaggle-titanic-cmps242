#Adapted from https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('../csv/train.csv', header=0)
test_df = pd.read_csv('../csv/test.csv', header=0)
ids = test_df['PassengerId'].values

def clean_data (df):
    clean_df = df
    clean_df = clean_gender(clean_df)
    clean_df = clean_ages(clean_df)
    clean_df = clean_embarked(clean_df)
    clean_df = clean_fares(clean_df)
    clean_df = clean_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    return clean_df.values

def clean_gender (df):
    df['Gender'] = df.Sex.map( {'female': 0, 'male': 1} ).astype(int)
    return df

def clean_ages (df):
    median_ages = np.zeros((2,3))
    for i in range(0,2):
        for j in range(0,3):
            median_ages[i,j] = df[(df.Gender == i) & \
                    (df.Pclass == j+1)]['Age'].dropna().median()

    for i in range(0,2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), \
                    'Age'] = median_ages[i,j]

    return df

def clean_embarked (df):
    if len(df.Embarked [ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(df.Embarked)))
    Ports_dict = {name: i for i, name in Ports }
    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
    return df

def clean_fares (df):
    #use stupid avg to fill in nulls
    avg_fare = df['Fare'].mean()
    df.loc[df.Fare.isnull(), 'Fare'] = avg_fare
    return df


train_data = clean_data(train_df)#.values
test_data = clean_data(test_df)#.values


print('Training...')
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

print('Predicting...')
output = forest.predict(test_data).astype(int)

print('Writing...')
predictions_file = open("firstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done.')
