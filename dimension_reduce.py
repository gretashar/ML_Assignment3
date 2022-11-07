from sklearn.decomposition import PCA, FastICA as ICA
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.preprocessing import StandardScaler , MinMaxScaler   
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from itertools import product
from collections import defaultdict
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.metrics.pairwise import pairwise_distances
from kmeans import kmeans_test
from NN import nn_test
from em import em_test
# def PCA(df):
#     pass

#https://www.kaggle.com/code/nirajvermafcb/principal-component-analysis-explained/notebook
def PCA_test(X,y,title):
    
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlim(0,7,1)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative explained variance')
    plt.grid()
    plt.savefig('./plots/'+title+' PCA.png')  
    plt.clf()




data = "./data/stroke.csv"
df = pd.read_csv(data)
df = df.replace('N/A', np.nan)
df = df.replace('Unknown', np.nan)
df = df.drop(['id'], axis=1)
df['avg_glucose_level'] = np.ceil((df['avg_glucose_level'].values))
df=df.dropna()
    
X = df.drop(["stroke"], axis=1)
# X=df
cols = X.columns
print(cols)


le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['ever_married'] = le.fit_transform(X['ever_married'])
X['work_type'] = le.fit_transform(X['work_type'])
X['Residence_type'] = le.fit_transform(X['Residence_type'])
X['avg_glucose_level'] = le.fit_transform(X['avg_glucose_level'])
X['smoking_status'] = le.fit_transform(X['smoking_status'])
# scaler =  MinMaxScaler()
scaler =  StandardScaler()
X = scaler.fit_transform(X)    
X = pd.DataFrame(X, columns=[cols])
y = df["stroke"]
correlation = X.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')
plt.savefig('./plots/'+"stroke"+' correlation.png')  
plt.clf()

PCA_test(X,y,"stroke")





# pca_stroke = PCA(n_components=5,random_state=2).fit_transform(X)
# ica_stroke = ICA(n_components=7,random_state=2).fit_transform(X)
# rca_stroke = RCA(n_components=6,random_state=2).fit_transform(X)

# pca_labels_km = kmeans_test(pca_stroke,y,'Stroke Prediction Data wPCA')
# ica_labels_km=kmeans_test(ica_stroke,y,'Stroke Prediction Data wICA')
# rca_labels_km=kmeans_test(rca_stroke,y,'Stroke Prediction Data wRCA')

# pca_labels_em = em_test(pca_stroke,y,'Stroke Prediction Data wPCA')
# ica_labels_em=em_test(ica_stroke,y,'Stroke Prediction Data wICA')
# rca_labels_em=em_test(rca_stroke,y,'Stroke Prediction Data wRCA')

    
data = "./data/heart.csv"
df = pd.read_csv(data)
  
X = df.drop(["output"], axis=1)
# X=df
cols = X.columns
scaler =  StandardScaler()
X = scaler.fit_transform(X)    
X = pd.DataFrame(X, columns=[cols])
y = df["output"]
correlation = X.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')
plt.savefig('./plots/'+"heart"+' correlation.png')  
plt.clf()


PCA_test(X,y,"heart")


pca_heart = PCA(n_components=5,random_state=2).fit_transform(X)
ica_heart = ICA(n_components=7,random_state=2).fit_transform(X)
rca_heart = RCA(n_components=6,random_state=2).fit_transform(X)

pca_labels_km = kmeans_test(pca_heart,y,'Heart Prediction Data wPCA')
ica_labels_km = kmeans_test(ica_heart,y,'Heart Prediction Data wICA')
rca_labels_km = kmeans_test(rca_heart,y,'Heart Prediction Data wRCA')

pca_labels_em = em_test(pca_heart,y,'Heart Prediction Data wPCA')
ica_labels_em=em_test(ica_heart,y,'Heart Prediction Data wICA')
rca_labels_em=em_test(rca_heart,y,'Heart Prediction Data wRCA')

print(pca_heart)
nn_test(pca_heart,y, "output", "Heart Failure Prediction with PCA")
nn_test(ica_heart,y, "output", "Heart Failure Prediction with ICA")
nn_test(rca_heart,y, "output", "Heart Failure Prediction with RCA")

print(pca_heart)
print(pca_labels_km)

