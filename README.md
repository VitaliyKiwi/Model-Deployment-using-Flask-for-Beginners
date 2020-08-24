# Model-Deployment-using-Flask-for-Beginners
This guide is for beginners to deploy a machine learning model using flask

# import pandas
import pandas as pd

# import numpy
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt

# import seaborn
import seaborn as sns

# import rcParams and set the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

# import KMeans from sklearn
from sklearn.cluster import KMeans

# read the data
df_customer = pd.read_csv("mall_customers.csv")

# print first five rows of data
df_customer.head()

# check the shape of the data
df_customer.shape

# check the data types of the variables
df_customer.dtypes

# check for missing values
df_customer.isnull().sum()

# drop unwanted column
df_customer.drop(['CustomerID'], axis=1, inplace=True)

# filter the numerical variables
df_num = df_customer.select_dtypes(include=np.number)

# print the first three rows of the data
df_num.head(3)

# create a histogram for numerical variables
df_num.hist()

# diaplay the plot
plt.show()

# create a boxplot for numeric variables
df_num.boxplot()

# create a countplot
sns.countplot(x='Genre', data=df_customer)
# display the plot
plt.show()

# create a barplot
df_customer.groupby('Genre')['Annual Income'].mean().plot(kind='bar')
# display the plot
plt.show()

# consider the variables
X = df_customer[['Annual Income', 'Spending Score (1-100)']]

# create empty dictionary
sse = {}
for k in range(1, 11): # select the range for k 
    kmeans = KMeans(n_clusters=k, random_state=42) # build the model
    kmeans.fit(X) # fit the model
    sse[k] = kmeans.inertia_ 
    
# set the label for x-axis
plt.xlabel('K')
# set the label for y-axis
plt.ylabel('Sum of Square Error(SSE)')
# plot the sse for different k values
plt.plot(list(sse.keys()), list(sse.values()))

# build model for k=5
model = KMeans(n_clusters=5, random_state=42)
# fit the model
model.fit(X)

# predict the values
y_predicted = model.fit_predict(X)

# add the new column to the dataframe
df_customer['cluster'] = y_predicted
# display the dataframe
df_customer.head()

# check the number of clusters
df_customer['cluster'].unique()

# get all the values
X = X.values

# Visualizing the clusters for k=5
plt.scatter(X[y_predicted==0,0],X[y_predicted==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(X[y_predicted==1,0],X[y_predicted==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(X[y_predicted==2,0],X[y_predicted==2,1],s=50, c='green',label='Cluster3')
plt.scatter(X[y_predicted==3,0],X[y_predicted==3,1],s=50, c='cyan',label='Cluster4')
plt.scatter(X[y_predicted==4,0],X[y_predicted==4,1],s=50, c='yellow',label='Cluster5')

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site')
plt.legend()
plt.show()

model.cluster_centers_

# check the data type of the 'cluster'
df_customer['cluster'].dtypes

# change the data type
df_customer = df_customer['cluster'].astype(object)

# serializing our model to a file called model.pkl
import pickle
pickle.dump(model, open("model.pkl","wb"))
