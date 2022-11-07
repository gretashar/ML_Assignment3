import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.preprocessing import StandardScaler , MinMaxScaler   
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score as sil_score, homogeneity_score
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder
from yellowbrick.cluster import KElbowVisualizer # cluster visualizer
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture as EM
import time

pd.set_option('display.max_columns', None)
# define dictionary for distance measures
#https://www.kaggle.com/code/arushchillar/kmeans-clustering-using-different-distance-metrics
# distance_measures = {'euclidean': 0, 'squared euclidean': 1, 'manhattan': 2, 'chebyshev': 3, 
#                     'canberra': 5, 'chi-square': 6}
distance_measures = {'euclidean': 0}

# function defined to compute purity score using pyclustering for various distance measures
def pyPurity(dist_measure,X,y,k):
    initial_centers = random_center_initializer(X, k, random_state=5).initialize()
    # instance created for respective distance metric
    instanceKm = kmeans(X, initial_centers=initial_centers, metric=distance_metric(dist_measure))
    # perform cluster analysis
    instanceKm.process()
    # cluster analysis results - clusters and centers
    pyClusters = instanceKm.get_clusters()
    pyCenters = instanceKm.get_centers()
    # enumerate encoding type to index labeling to get labels
    pyEncoding = instanceKm.get_cluster_encoding()
    pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)
    pyLabels = pyEncoder.set_encoding(0).get_clusters()
    # function purity score is defined in previous section
    return purity_score(y, pyLabels)

#https://www.kaggle.com/code/arushchillar/kmeans-clustering-using-different-distance-metrics
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    confusion_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)

def em_test(X,y, title):
    
    k=5
    # if title == "Stroke Prediction":
    #     df = df.replace('N/A', np.nan)
    #     df = df.replace('Unknown', np.nan)
    #     df = df.drop(['id'], axis=1)
    #     df['avg_glucose_level'] = np.ceil((df['avg_glucose_level'].values))
    #     df=df.dropna()
    #     # k=7

    
    # X = df.drop([output], axis=1)
    # # X=df
    # cols = X.columns
    # print(cols)
    
    # if title == "Stroke Prediction":
    #     le = LabelEncoder()
    #     X['gender'] = le.fit_transform(X['gender'])
    #     X['ever_married'] = le.fit_transform(X['ever_married'])
    #     X['work_type'] = le.fit_transform(X['work_type'])
    #     X['Residence_type'] = le.fit_transform(X['Residence_type'])
    #     X['avg_glucose_level'] = le.fit_transform(X['avg_glucose_level'])
    #     X['smoking_status'] = le.fit_transform(X['smoking_status'])
    # scaler =  MinMaxScaler()
    # # scaler =  StandardScaler()
    # X = scaler.fit_transform(X)    
    # X = pd.DataFrame(X, columns=[cols])
    # y = df[output]
        
    # print(X.head())
    # print(title)
    kmeans = KMeans(n_clusters=2, random_state=0) 
    kmeans.fit(X)
    
    labels = kmeans.labels_

    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)

    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    
    cs = []
    times=[]
    for i in range(2, 50):
        em = EM(n_components = i, covariance_type='diag',n_init=1,warm_start=True, random_state = 0)
        start = time.time()
        em.fit(X)
        end = time.time()
        train_time = end-start
        labels =  em.predict(X)
        cs.append(sil_score(X, labels))
        # cs.append(em.inertia_)
        times.append(train_time)
    
        
    plt.plot(range(2, 50), cs)
    plt.title('Avg Silhouette Score '+title)
    plt.xlabel('Number of clusters')
    plt.grid()
    plt.ylabel('Avg Sil Score')
    plt.savefig('./plots/'+title+' em_elbow_plot.png')  
    plt.clf()
    
    plt.plot(range(2, 50), times)
    plt.title('Cluster vs Time for '+title)
    plt.xlabel('Number of clusters')
    plt.ylabel('Time (s)')
    plt.grid()
    plt.savefig('./plots/'+title+' em_time_plot.png')  
    plt.clf()
        
        
    
    # Instantiate the clustering model and visualizer
    # model = em()
    # # model = em = EM()
    # visualizer = KElbowVisualizer(model, k=(1, 25))

    # visualizer.fit(X) # Fit the data to the visualizer
    # # visualizer.show() # Finalize and render the figure
    
    # plt.savefig('./plots/'+title+' time_plot.png')  
    # plt.clf()
    
    # # instatiate em class and set the number of clusters
    # km_model = em(n_clusters=k, random_state=10)

    # # call fit method with data 
    # km = km_model.fit_predict(X)

    # # coordinates of cluster center
    # centroids = km_model.cluster_centers_ 

    # # cluster label for each data point
    # labels = km_model.labels_ 
    # # Report Purity Score
    # purity = purity_score(y, labels)
    # print(f"The purity score is {round(purity*100, 2)}%")
    # plt.show()
    
    # print results
    # for measure, value in distance_measures.items():
    #     print(f"The purity score for {measure} distance is {round(pyPurity(value,X,y,k)*100, 2)}%")


    # initial_centers = random_center_initializer(X, k, random_state=5).initialize()
    # instance created for respective distance metric
    # instanceKm = em(X, initial_centers=initial_centers, metric=distance_metric(0))
    # perform cluster analysis
    # instanceKm.process()
    # # cluster analysis results - clusters and centers
    # pyClusters = instanceKm.get_clusters()
    # pyCenters = instanceKm.get_centers()
    # # enumerate encoding type to index labeling to get labels
    # pyEncoding = instanceKm.get_cluster_encoding()
    # pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)
    # pyLabels = pyEncoder.set_encoding(0).get_clusters()


data = "./data/heart.csv"
df = pd.read_csv(data)

X = df.drop(["output"], axis=1)
# X=df
cols = X.columns
# scaler =  StandardScaler()
scaler =  MinMaxScaler()
X = scaler.fit_transform(X)    
X = pd.DataFrame(X, columns=[cols])
y = df["output"]
em_test(X, y, "Heart Failure Prediction")


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
scaler =  MinMaxScaler()
# scaler =  StandardScaler()
X = scaler.fit_transform(X)    
X = pd.DataFrame(X, columns=[cols])
y = df["stroke"]
em_test(X, y,"Stroke Prediction")

