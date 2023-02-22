# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:52:59 2023

@author: User
"""
#importing the relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import err_ranges as err

#fuction definition
"""

def solution(filename,countries,columns,indicator):
    df = pd.read_csv(filename,skiprows=4)
    df = df[df['Indicator Name'] == indicator]
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    df = df.loc[countries]
    
    
"""
#gives the csv file and the selected indicator
url = 'API_19_DS2_en_csv_v2_4700503.csv'
indicator = 'Population growth (annual %)'
year1 = '1980'
year2 = '2020'

#reads in the csv file from the url with the selected indicator while skiping the first four rows not needed
df = pd.read_csv(url, skiprows=4)
df = df.loc[df['Indicator Name'] == indicator]

#creates dataframe with two columns for clustering
df_cluster = df.loc[df.index, [ 'Country Name', year1, year2]].dropna()

#get the dataframe of year1 ans year2 to an array
x = df_cluster[[year1, year2]].values

#this shows the countries with higher population growth
df_cluster = df_cluster.sort_values(year1, ascending=False)
print(df_cluster.head(10))

#visualise the data
plt.figure()
df_cluster.plot(year1, year2, kind='scatter')
plt.show()

sse = []
means = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)

# Generate the Elbow Plot
plt.plot(range(1, 11), sse)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Error [Inertia]")
plt.savefig('clusters.png')
plt.show()

# sets up the clusterer with the number of expected clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# fits the data, while the results are stored in the kmeans object
y_kmeans = kmeans.fit_predict(x)
# prints the labels of the associated clustersof x_predict
labels=kmeans.labels_
print(labels)

#gets the size of the of the cluster and the predicted x
print(y_kmeans.shape)
print(df_cluster.shape)

#this creates a new dataframe with the labels for each country
df_cluster['label'] = y_kmeans
df_label = df_cluster.loc[df_cluster['label'] == 1]
print(df_label.head(10))

#extract the estimated cluster centres
y = kmeans.cluster_centers_
print(y)

# shows the cluster centres
c = kmeans.cluster_centers_
print(c)

#calculate the silhoutte score
print(skmet.silhouette_score(x, labels))

#plots the clusters and the centroids
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 30, color = 'olive',label = 'cluster 0')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 30, color = 'cyan',label = 'cluster 1')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 30, color= 'purple',label = 'cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 10, color = 'red', label = 'Centroids')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# function for fitting

def model(x, a, b, c, d):
    '''
    
    docstring
    
    '''
    x = x - 2000
    
    return a*x**3 + b*x**2 + c*x + d

#transposes the data
df_year = df.T

#rename the columns
df_year = df_year.rename(columns=df_year.iloc[0])

#drop the country name
df_year = df_year.drop(index=df_year.index[0], axis=0)

#extracts the year index
df_year['Year'] = df_year.index
#shows the dataframe
print(df_year)

#converts the dataframe to numeric values
df_fitting = df_year[['Year', 'India']].apply(pd.to_numeric, errors='coerce')

#ddrops the null values
m = df_fitting.dropna().values

#extracts the x-dimension and the y-dimension
x_d, y_d = m[:,0], m[:,1]
 
#fits the model
popt, a = opt.curve_fit(model, x_d, y_d)
param, covar = opt.curve_fit(model,  x_d, y_d)
a, b, c, d = popt
print(a, b, c)

#prints the error
sigma = np.sqrt(np.diag(covar))
print(sigma)
#calculates the lower and upper limit of the function
low, up = err.err_ranges( x_d, model, popt,sigma)
#plots the curve
plt.scatter(x_d, y_d)

x_line = np.arange(min(m[:,0]), max(m[:,0])+1, 1)
y_line = model(x_line, a, b, c, d)

#plots the fit
plt.scatter(x_d, y_d)
plt.plot(x_line, y_line, '--', color='black')
plt.fill_between(x_d, low, up, alpha=0.7, color='cyan')
plt.xlabel("Year")
plt.ylabel("Population growth")
plt.legend()
plt.show()


