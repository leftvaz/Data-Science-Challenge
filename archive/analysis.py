# import libraries
import pandas as pd
import glob
import os
import datetime
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

# set path where files are located 
# (used \\ because it is a windows environment and \ is an escape character)
path = 'C:\\Users\\lefte\\Desktop\\Data Science Challenge\\all journeys'
path = path.replace('\\', os.sep)

# read each file, while assigning to it a journey id
journeys = pd.DataFrame()
i = 0

for filename in glob.glob(os.path.join(path, '*.csv')):
    print(filename)
    current_journey = pd.read_csv(filename)
    current_journey = current_journey.rename(columns=lambda x: x.strip())
    current_journey['Journey_ID'] = i
    journeys = journeys.append(current_journey)
    i += 1
    
journeys.sample(5)

# get timestamp in readable format
def dt_conv(x):
    s = x/1000.0
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
    
journeys['timestamp'] = journeys['timestamp'].apply(lambda x: dt_conv(x))
journeys.head(10)

# fix accelerometer unit of measure 
# a fast car can reach an acceleration of around 6 to 6.5 m/s2
# there are also speed limits to take into account
# any max acceleration values of over 5 are assumed to be in m/s2
# similarly, decelaration is affected by speed

condition = (journeys.loc[:, ('Journey_ID', 'x', 'y', 'z')].groupby(['Journey_ID']).max().max(1)) > 5
acc_in_ms2 = condition[condition == True].index
condition2 = journeys['Journey_ID'].isin(acc_in_ms2)

journeys.loc[condition2, 'x'] = journeys.loc[condition2, 'x'].apply(lambda x: x/9.81)
journeys.loc[condition2, 'y'] = journeys.loc[condition2, 'y'].apply(lambda x: x/9.81)
journeys.loc[condition2, 'z'] = journeys.loc[condition2, 'z'].apply(lambda x: x/9.81)


# fill na with previous value (to fix empty gps and empty accelerometer)
journeys.fillna(method = 'ffill', inplace = True)
journeys.fillna(method = 'bfill', inplace = True)



# create accelerometer value regardless of axis (the absolute max of x,y, z at any given time)
maxCol=lambda x: max(x.min(), x.max(), key=abs)
journeys['acceleration'] = journeys[['x', 'y', 'z']].apply(maxCol,axis=1)




### preprocessing over, moving on to algorithm (anomaly detection)

#################
# One Class SVM #
#################
outliers_fraction = 0.001
min_max_scaler = preprocessing.StandardScaler()

df = journeys.drop(['type', 'timestamp'], axis = 1)
data = df
np_scaled = min_max_scaler.fit_transform(data)
# train one class SVM 
model =  OneClassSVM(nu=0.95 * outliers_fraction + 0.05) #nu=0.95 * outliers_fraction  + 0.05
data = pd.DataFrame(np_scaled)
model.fit(data)
# add the data to the main  
df['anomaly26'] = pd.Series(model.predict(data))
df['anomaly26'] = df['anomaly26'].map( {1: 0, -1: 1} )
print(df['anomaly26'].value_counts())








# test
journeys[(journeys.Journey_ID == 21)].plot(x = 'timestamp', y = 'acceleration')
# probably a crash since it is a long period of immobility followed by slow movement

outliers_fraction = 0.0001
min_max_scaler = preprocessing.StandardScaler()

df = journeys[(journeys.Journey_ID == 14)]
data = journeys[(journeys.Journey_ID == 14)].loc[:, ('acceleration', 'speed')]
np_scaled = min_max_scaler.fit_transform(data)
# train one class SVM 
model = IsolationForest(contamination = outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
data = pd.DataFrame(np_scaled)
model.fit(data)
# add the data to the main  
df['anomaly26'] = pd.Series(model.predict(data))
df['anomaly26'] = df['anomaly26'].map( {1: 0, -1: 1} )
print(df['anomaly26'].value_counts())

fig, ax = plt.subplots()

a = df.loc[df['anomaly26'] == 1, ['timestamp', 'acceleration']] #anomaly

ax.plot(df['timestamp'], df['acceleration'], color='blue')
ax.scatter(a['timestamp'],a['acceleration'], color='red')
plt.show()


