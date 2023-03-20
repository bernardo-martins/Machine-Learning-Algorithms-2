# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:14:15 2020

@author: Tiago Botelho 52009 Bernardo Martins 53292

"""
import numpy as np
import os.path as os
import tp2_aux as aux
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pandas.plotting import scatter_matrix


def load_images():
    images = aux.images_as_matrix()

    targets = np.loadtxt('labels.txt', delimiter=',', dtype=int)
    non_zero_indexs = [i for i in range(len(targets[:, 1])) if targets[:, 1][i] > 0]
    zero_indexs = [i for i in range(len(targets[:, 1])) if targets[:, 1][i] == 0]
    non_zero_images = images[non_zero_indexs]

    return non_zero_images, images, targets[non_zero_indexs][:, 1], targets[:, 1], non_zero_indexs, zero_indexs


def calc_metrics(features, predicts, labels):
    tp = 0 
    tn = 0 
    fp = 0 
    fn = 0

    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            if (i > j):
                if (labels[i] == labels[j]):
                    if (predicts[i] == predicts[j]):
                        tp += 1
                    else:
                        fn += 1
                if (labels[i] != labels[j]):
                    if (predicts[i] == predicts[j]):
                        fp += 1
                    else:
                        tn += 1
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f1 = 2 * ((precision * recall) / (precision + recall))
    rand = (tp + tn)/((labels.shape[0]*(labels.shape[0]-1))/2)
    if(len(np.unique(predicts))<2):
      return precision, recall, f1, rand, 0, 0

    silh_score = silhouette_score(features, predicts)
    adj_rand_score = adjusted_rand_score(labels, predicts)

    return precision, recall, f1, rand, silh_score, adj_rand_score


def std_data(data):
    means = np.mean(data, axis=0)
    stdevs = np.std(data, axis=0)

    for i in range(len(stdevs)):
        if (np.all((stdevs[i] == 0))):
            stdevs[i] = stdevs[i] + 1
    data = (data - means) / stdevs

    return data


def normalize(data):

    max = np.max(data, axis = 0)
    min = np.min(data, axis = 0)

    div = max-min

    for i in range(len(div)):
          if (np.all((div[i] == 0))):
              div[i] = div[i] + 1
    data = (data-min)/(div)
    
    return data



def PCAAnalysis(data):
    # computar a matriz de covariancia (2500*2500 matrix) e os eigenvalues sao calculados a partir
    # matrix. Ver quais são os mais elevados e multiplicar pelos dados, selecionar as colunas cujos eigenvalues sãoão
    # os mais elevados. isto é tudo feito pelo pca.fit_transform
    if (not os.isfile('pca.npy')):
        pca = PCA(n_components=6)
        pcaPrincComp = pca.fit_transform(data)
        np.save('pca', pcaPrincComp)
    else:
        pcaPrincComp = np.load('pca.npy')
    return pcaPrincComp


def tSNE(data):
    # Utiliza a probabilidade numa distribuição Student de um vértice xi ser vizinho de um vértice xj
    # e minimiza a divergência entre pontos. Isto é feito por substituição da distribuição gaussiana
    if (not os.isfile('tsne.npy')):
        tsne = TSNE(n_components=6, method='exact')
        tsnePrincComp = tsne.fit_transform(data)
        np.save('tsne', tsnePrincComp)
    else:
        tsnePrincComp = np.load('tsne.npy')

    return tsnePrincComp

def isoMap(data):

    # cria uma matriz de distancias entre pontos, computa no modelo lower-dimensional vários
    # embeddings para ver qual é o que minimiza a distancia entre pontos e escolhe esse embedding
    if (not os.isfile('isoMap.npy')):
        isoMap = Isomap(n_components=6)
        isoMapPrincComp = isoMap.fit_transform(data)
        np.save('isoMap', isoMapPrincComp)
    else:
        isoMapPrincComp = np.load('isoMap.npy')

    return isoMapPrincComp


def extractFeatures(data):
    data1 = PCAAnalysis(normalize(data))
    # utiliza a distribuição e distancia entre pontos portanto convém standardizar
    data2 = tSNE(data)
    data3 = isoMap(data)
    pdFinal = pd.concat(
        [pd.DataFrame(data=data1, columns=[0, 1, 2, 3, 4, 5]), pd.DataFrame(data=data2, columns=[6, 7, 8, 9, 10, 11]),
         pd.DataFrame(data=data3, columns=[12, 13, 14, 15, 16, 17])], axis=1)
    
    return pdFinal


def select_best_features(features, targets, threshold):
    f, prob = f_classif(features, targets)
    plt.plot(f, 'x')
    plt.show()
    k = [i for i in range(len(f)) if f[i] > threshold]

    return k


def kMeansClassif(k, features_tr, features_test, labels, labeled_indexs):
    kmeans = KMeans(n_clusters = k, random_state=0, max_iter=150)
    kmeans.fit(features_tr)

    predicts = kmeans.predict(features_test)
    aux.report_clusters(np.arange(len(predicts)), predicts, "kMeans.html")

    return calc_metrics(features_test[labeled_indexs, ] ,predicts[labeled_indexs], labels[labeled_indexs])


def plot_statistics(data):
    dataf = pd.DataFrame(data=data)

    dataf.plot.box()
    plt.ylim((-100,100))
    plt.savefig('box.png', alpha=0.5, figsize=(15, 10))
    plt.show()
    plt.close()

    scatter_matrix(dataf, alpha=0.5, figsize=(15, 10) , diagonal='kde')
    plt.savefig('scatter_matrix.png', dpi=200, bbox_inches='tight')
    #plt.show()
    plt.close() 

    return


def dbscan(labeled_final_features, features_selected, targets, labeled_indexs):

    knn = KNeighborsClassifier(n_neighbors = 5).fit(features_selected, targets).kneighbors(return_distance = True)

    dists = knn[0][:,4]
    sorted_dists = np.sort(dists)
    plt.title('Epsílon Var')
    plt.plot( np.sort(dists)[::-1])
    plt.show()

    precision_arr = []
    recall_arr = []
    f1_arr = []
    rand_arr = []
    silhscore_arr = []
    adj_rand_score_arr = []

    for e in sorted_dists:
      dbscan = DBSCAN(eps=e, min_samples=5, metric='euclidean').fit(features_selected)
      precision, recall, f1, rand, silh_score, adj_rand_score =  calc_metrics(features_selected, dbscan.labels_, targets)
      precision_arr.append(precision)
      recall_arr.append(recall)
      f1_arr.append(f1)
      rand_arr.append(rand)
      silhscore_arr.append(silh_score)
      adj_rand_score_arr.append(adj_rand_score)
      


    dbscan = DBSCAN(eps = 0.4, min_samples=5, metric='euclidean').fit(features_selected)
    aux.report_clusters(np.arange(len(dbscan.labels_)), dbscan.labels_, "dbscan.html")

    plt.plot(sorted_dists,precision_arr, label = "Precision")
    plt.plot(sorted_dists,recall_arr, label = "Recall")
    plt.plot(sorted_dists, f1_arr, label = 'F1')
    plt.plot(sorted_dists,rand_arr, label = 'Rand Score')
    plt.plot(sorted_dists, silhscore_arr, label = 'Silhouette_avg')
    plt.plot(sorted_dists, adj_rand_score_arr, label = 'Adjusted Rand Score')
    plt.legend()
    plt.title('DBSCAN')
    plt.show()
     
    return


def agglomerative_clust(labeled_final_features, features_selected, targets, labeled_indexs):

    cluster_space = np.arange(3,15,1)
    precision_arr = []
    recall_arr = []
    f1_arr = []
    rand_arr = []
    silhscore_arr = []
    adj_rand_score_arr = []

    for c in cluster_space:

      connectivity_graph = kneighbors_graph(features_selected, c, mode='connectivity', include_self=True)
      agglomerative = AgglomerativeClustering(n_clusters = c,  linkage = "ward", connectivity=connectivity_graph)
      agglomerative = agglomerative.fit(features_selected)
      precision, recall, f1, rand, silh_score, adj_rand_score =  calc_metrics(features_selected, agglomerative.labels_, targets)
      precision_arr.append(precision)
      recall_arr.append(recall)
      f1_arr.append(f1)
      rand_arr.append(rand)
      silhscore_arr.append(silh_score)
      adj_rand_score_arr.append(adj_rand_score)

    connectivity_graph = kneighbors_graph(features_selected, 7, mode='connectivity', include_self=True)
    agglomerative = AgglomerativeClustering(n_clusters = 7,  linkage = "ward", connectivity=connectivity_graph).fit(features_selected)

    aux.report_clusters(np.arange(len(agglomerative.labels_)), agglomerative.labels_, "agglomerative.html")
    plt.plot(cluster_space,precision_arr, label = "Precision")
    plt.plot(cluster_space,recall_arr, label = "Recall")
    plt.plot(cluster_space, f1_arr, label = 'F1')
    plt.plot(cluster_space,rand_arr, label = 'Rand Score')
    plt.plot(cluster_space, silhscore_arr, label = 'Silhouette_avg')
    plt.plot(cluster_space, adj_rand_score_arr, label = 'Adjusted Rand Score')
    plt.legend()
    plt.title('Agglomerative Clustering')
    plt.show()
    plt.close()

    return


def bissecting_kmeans(labeled_final_features, features_selected, targets, labeled_indexs):
    divisions = 5
    matrix = np.zeros((divisions, 2))
    algs = []
    data = np.array(labeled_final_features)
    for i in range(divisions):
      kmeans = KMeans(n_clusters = 2, random_state=0, max_iter=150).fit(data)
      results = kmeans.labels_
      count_non_zero = np.count_nonzero(results==0)
      algs.append(kmeans)
      if( count_non_zero > len(results) - count_non_zero ):
        non_zero_indexs = [i for i in range(len(results)) if results[i] > 0]
        data = data[non_zero_indexs,:]
        matrix[i][1] = 1
      else:
        zero_indexs = [i for i in range(len(results)) if results[i] == 0]
        data = data[zero_indexs, :]
        matrix[i][0] = 1
    
    results = []
    
    for example in features_selected:
      var = []
      for i in range(len(algs)):
        prediction = algs[i].predict([example])
        if ( prediction[0] == [1] ):
          var.append(1)
          if ( matrix[i][0] ):

            break
        else:
          var.append(0)
          if ( matrix[i][1] ):
            
            break
      results.append(var)
    
    aux.report_clusters_hierarchical(np.arange(0,features_selected.shape[0]), results, "bissecting_k_means.html")

    return

def get_data():

    labeled_features, features, labeled_targets, targets, labeled_indexs, zero_indexs = load_images()

    all_features = extractFeatures(features)  # 18 features

    all_features = all_features.drop([11, 12, 13, 14, 16], axis=1).values
    
    labeled_features = all_features[labeled_indexs]

    threshold = 10
    
    best_columns_indexs = select_best_features(labeled_features, labeled_targets, threshold)

    features_selected = all_features[:, best_columns_indexs]

    plot_statistics(all_features)

    features_selected = std_data(features_selected)

    labeled_final_features = features_selected[labeled_indexs]

    return labeled_final_features, features_selected, targets, labeled_indexs

def k_means(labeled_final_features, features_selected, targets, labeled_indexs):

    precision_arr = []
    recall_arr = []
    f1_arr = []
    rand_arr = []
    silhscore_arr = []
    adj_rand_score_arr = []

    cluster_space = np.arange(3,15,1)
    for k in cluster_space:

        precision, recall, f1, rand, silh_score, adj_rand_score = kMeansClassif(k, labeled_final_features, features_selected, targets, labeled_indexs)

        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)
        rand_arr.append(rand)
        silhscore_arr.append(silh_score)
        adj_rand_score_arr.append(adj_rand_score)
        

    precision, recall, f1, rand,silh_score, adj_rand_score = kMeansClassif(8, labeled_final_features, features_selected, targets, labeled_indexs)
    
    plt.plot(cluster_space,precision_arr, label = "Precision")
    plt.plot(cluster_space,recall_arr, label = "Recall")
    plt.plot(cluster_space, f1_arr, label = 'F1')
    plt.plot(cluster_space,rand_arr, label = 'Rand Score')
    plt.plot(cluster_space, silhscore_arr, label = 'Silhouette_avg')
    plt.plot(cluster_space, adj_rand_score_arr, label = 'Adjusted Rand Score')
    plt.legend()
    plt.title('K-Means Clustering')
    plt.show()

    return

def _main():

  labeled_final_features, features_selected, targets, labeled_indexs = get_data()

  print('K-Means') 
  k_means(labeled_final_features, features_selected, targets, labeled_indexs)
  print('---------------------')
  print('DBSCAN')
  dbscan(labeled_final_features, features_selected, targets, labeled_indexs)
  print('---------------------')
  print('Agglomerative Clustering')
  agglomerative_clust(labeled_final_features, features_selected, targets, labeled_indexs)
  print('---------------------')
  print('Bissecting K-Means')
  bissecting_kmeans(labeled_final_features, features_selected, targets, labeled_indexs)


_main()
