#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A kernel density estimation based entropic ISOMAP for metric learning

Python source code for KDE-based ISOMAP 

Discussion about bandwidth estimation rules:
https://stats.stackexchange.com/questions/90656/kernel-bandwidth-scotts-vs-silvermans-rules

https://stats.stackexchange.com/questions/90656/kernel-bandwidth-scotts-vs-silvermans-rules

https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width

"""

# Imports
import time
import warnings
import umap
import numpy as np
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph_shortest_path as sksp
import scipy as sp
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from scipy.stats import iqr
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.nonparametric.bandwidths import bw_scott
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

#%%%%%%%%%%%%% Functions
'''
Computes the symmetrized KL-divergence (relative entropy) between
2 densities in a non-parametric way
'''
def divergenciaKL(dens1, dens2):
    k1 = len(dens1)
    k2 = len(dens2)
    
    # Remove zeros and too small values
    dens1[dens1<10**(-300)] = 10**(-300)
    dens2[dens2<10**(-300)] = 10**(-300)
    
    if k1 != k2:
        return -1
    else:
        dKL12 = sum(dens1*np.log(dens1/dens2))/k1
        dKL21 = sum(dens2*np.log(dens2/dens1))/k2
        dKL = 0.5*(dKL12 + dKL21)
        
        return dKL
    
'''
Estimation of h (bandwidth) through Silverman method (rule of thumb)
https://en.wikipedia.org/wiki/Kernel_density_estimation
'''
def Silverman(dados):    
    num = len(dados)
    
    dp = dados.std()    
    inter = iqr(dados)/1.34
    hs = 0.9*min(dp, inter)*num**(-0.2)
    # h cannot be zero, nor too close to zero
    hs = max(hs, 0.05)

    return hs

'''
Estimation of h (bandwidth) through Scott
https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
'''
def Scott(dados):    
    num = len(dados)
    
    dp = dados.std()    
    hs = 3.49*dp*num**(-1/3)
    # hs cannot be zero, nor too close to zero
    hs = max(hs, 0.05)

    return hs

'''
Estimation of h (bandwidth) through Freedman and Diaconis
https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
'''
def Freedman_Diaconis(dados):    
    num = len(dados)
    
    dp = dados.std()    
    
    hfd = 2*iqr(dados)*n**(-1/3)
    hfd = max(hfd, 0.05)

    return hfd


'''
 Performs a grid search cross-validation to optimize the value h
'''
def CrossValidation(dados):    
    # Search in the interval [0, 1] (20 points) 
    params = {'bandwidth': np.linspace(0, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(dados)
    # h cannot be zero, nor too close to zero
    hcv = max(grid.best_estimator_.bandwidth, 0.05)
    
    return hcv

'''
Non-parametric estimation of the local densities 
Parameters:
 dados: data matrix
 A: adjacency matrix of the KNN graph
 method: string that controls the bandwidth estimation
               'none, silverman, silverman/3, scott, crossval, isj'
'''
def KernelDensityEstimation(dados, A, method, h=0.1, delta=256):
    # Number of points
    n = dados.shape[0]
    # Number of features
    m = dados.shape[1]
    
    # 3D matrix used to store the local densities
    # n matrices of dimensions delta x m
    densidades = np.zeros((n, delta))

    # Determine the minimum and maximum value in all features
    lista = []
    for i in range(m):
        F = dados[:, i]
        lista.append(F.min())
        lista.append(F.max())
    
    # We adjust the x axis so that all local densities are plotted in the same interval
    minimo = min(lista)
    maximo = max(lista)

    # Non-parametric estimation of the local densities
    for i in range(n):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = dados[indices]
        
        F = amostras
    
        X = F.reshape(-1, 1)
        
        if method == 'isj':
            # Bandwidth estimation through Improved Sheather-Jones
            # More details in: https://arxiv.org/pdf/1011.2602.pdf
            kde = FFTKDE(kernel='gaussian', bw='ISJ')
            kde.fit(X)

            x_plot = np.linspace(minimo-0.1, maximo+0.1, delta)
            pontos = x_plot.reshape(-1, 1)

            dens = kde.evaluate(pontos)
      
        elif method == 'crossval':    
            # Cross-Validation rule
            h = CrossValidation(X)
                    
            kde = KernelDensity(kernel='gaussian', bandwidth=h) 
            kde.fit(X)

            x_plot = np.linspace(minimo, maximo, delta)
            pontos = x_plot.reshape(-1, 1)

            log_dens = kde.score_samples(pontos)
            dens = np.exp(log_dens)
        
        elif method == 'silverman':
            # Silverman
            h = Silverman(X)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=h) 
            kde.fit(X)

            x_plot = np.linspace(minimo, maximo, delta)
            pontos = x_plot.reshape(-1, 1)

            log_dens = kde.score_samples(pontos)
            dens = np.exp(log_dens)
            
        elif method == 'silverman/3':
            # Silverman
            h = Silverman(X)/3.0
            
            kde = KernelDensity(kernel='gaussian', bandwidth=h) 
            kde.fit(X)

            x_plot = np.linspace(minimo, maximo, delta)
            pontos = x_plot.reshape(-1, 1)

            log_dens = kde.score_samples(pontos)
            dens = np.exp(log_dens)
            
        elif method == 'FD':
            # Silverman
            h = Freedman_Diaconis(X)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=h) 
            kde.fit(X)

            x_plot = np.linspace(minimo, maximo, delta)
            pontos = x_plot.reshape(-1, 1)

            log_dens = kde.score_samples(pontos)
            dens = np.exp(log_dens)

        elif method == 'scott':

            # Scott
            h = Scott(X)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=h) 
            kde.fit(X)

            x_plot = np.linspace(minimo, maximo, delta)
            pontos = x_plot.reshape(-1, 1)

            log_dens = kde.score_samples(pontos)
            dens = np.exp(log_dens)
            
        else: # none (constant h, the same for all densities)
        
            kde = KernelDensity(kernel='gaussian', bandwidth=h) 
            kde.fit(X)

            x_plot = np.linspace(minimo, maximo, delta)
            pontos = x_plot.reshape(-1, 1)

            log_dens = kde.score_samples(pontos)
            dens = np.exp(log_dens)
        
        densidades[i, :] = dens
            
    return densidades

'''
 Non-parametric PCAKL with Kernel density estimation
 dados: data matrix
 delta: number of points in each density (default = 256)
 k: number of neighbors in the KNN graph (patch size)
         (this parameter has strong influence in the results)
'''
def NonParamISO(dados, p, d):
    # Number of features
    m = dados.shape[1]
    # Number of samples
    n = dados.shape[0]

    # Creates a KNN graph from the dataset (the value of K affects the results )
    # The second parameter is the number of neighbors K
    esparsa = sknn.kneighbors_graph(dados, n, mode='distance', include_self=True)
    A = esparsa.toarray()

    # Calcula a distância média de xi a cada outro vértice
    #avg_distances = A.mean(axis=1)
    #desvpad = A.std(axis=1)
    #med_distances = np.median(A, axis=1)
    percentiles = np.percentile(A, p, axis=1)
    # Se distância entre xi e xj é maior que a média, desconecta do grafo
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] > percentiles[i]:
                A[i, j] = 0
            else:
                A[i, j] = 1
    
    # silverman, scott, none (default: h = 0.1), crossval
    densidades = KernelDensityEstimation(dados, A, 'scott')

    # Matriz de pesos inicialmente igual a A
    W = A.copy()
    # Define the vector of KL-divergences
    #vetor_dKL = np.zeros(m)
    # Computes the weights using de KL divergence
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            dKL = divergenciaKL(densidades[i, :], densidades[j, :])     # Assim não funciona bem
            if W[i, j] > 0:
                W[i, j] = dKL
                #W[i, j] = vetor_dKL.sum()

    # Computes geodesic distances in W
    D = sksp.graph_shortest_path(W, directed=False)
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

'''
 Regular PCA implementation
'''
def myPCA(dados):
    # Eigenvalues and eigenvectors of the covariance matrix
    v1, w1 = np.linalg.eig(np.cov(dados.T))

    # Sort the eigenvalues
    ordem = v1.argsort()

    # Select the two eigenvectors associated to the two largest eigenvalues
    maior_autovetor1 = w1[:, ordem[-1]]
    segundo_maior1 = w1[:, ordem[-2]]

    # Projection matrix
    Wpca = np.array([maior_autovetor1, segundo_maior1])

    # Linear projection into the 2D subspace
    novos_dados_pca = np.dot(Wpca, dados.T)

    return novos_dados_pca

'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    print('KNN accuracy: ', acc)

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc = svm.score(X_test, y_test)
    lista.append(acc)
    print('SVM accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    print('QDA accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    #average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Maximum accuracy: ', maximo)
    print()

    return [sc, maximo]
    

#%%%%%%%%%%%%%%%%%  Beginning of the script

# Datasets
#X = skdata.load_iris()
#X = skdata.load_wine()   
#X = skdata.fetch_openml(name='Engine1', version=1) 
#X = skdata.fetch_openml(name='prnn_crabs', version=1) 
#X = skdata.fetch_openml(name='analcatdata_happiness', version=1) 
X = skdata.fetch_openml(name='mux6', version=1) 
#X = skdata.fetch_openml(name='parity5', version=1) 
#X = skdata.fetch_openml(name='vertebra-column', version=1) 
#X = skdata.fetch_openml(name='hayes-roth', version=2)  
#X = skdata.fetch_openml(name='aids', version=1) 
#X = skdata.fetch_openml(name='pm10', version=2) 
#X = skdata.fetch_openml(name='strikes', version=2)  
#X = skdata.fetch_openml(name='disclosure_z', version=2) 
#X = skdata.fetch_openml(name='diggle_table_a2', version=2) 
#X = skdata.fetch_openml(name='monks-problems-1', version=1) 
#X = skdata.fetch_openml(name='breast-tissue', version=2) 
#X = skdata.fetch_openml(name='planning-relax', version=1) 
#X = skdata.fetch_openml(name='haberman', version=1) 
#X = skdata.fetch_openml(name='rmftsa_ladata', version=2) 
#X = skdata.fetch_openml(name='KnuggetChase3', version=1) 
#X = skdata.fetch_openml(name='bolts', version=2) 
#X = skdata.fetch_openml(name='fl2000', version=2) 
#X = skdata.fetch_openml(name='triazines', version=2) 
#X = skdata.fetch_openml(name='fri_c2_100_10', version=2) 
#X = skdata.fetch_openml(name='Touch2', version=1) 
#X = skdata.fetch_openml(name='veteran', version=2) 
#X = skdata.fetch_openml(name='vineyard', version=2) 
#X = skdata.fetch_openml(name='diabetes_numeric', version=2) 
#X = skdata.fetch_openml(name='prnn_fglass', version=2) 
#X = skdata.fetch_openml(name='parkinsons', version=1) 
#X = skdata.fetch_openml(name='acute-inflammations', version=2) 
#X = skdata.fetch_openml(name='blogger', version=1) 
#X = skdata.fetch_openml(name='prnn_viruses', version=1) 
#X = skdata.fetch_openml(name='analcatdata_creditscore', version=1) 
#X = skdata.fetch_openml(name='zoo', version=1) 
#X = skdata.fetch_openml(name='confidence', version=2) 

#X = skdata.fetch_openml(name='rabe_131', version=2)
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='collins', version=2)
#X = skdata.fetch_openml(name='MindCave2', version=1) 

dados = X['data']
target = X['target']  

# Number of samples
n = dados.shape[0]
print('Number of samples: %d' %n)
# Number of features
m = dados.shape[1]
print('Number of features: %d' %m)
# Number of classes
c = len(np.unique(target))
print('Number of classes: %d' %c)
print()

#%%%%%%%%%%%%%%%%%%%%%% Data processing

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)


#%%%%%%%%%%% Simple PCA 
dados_pca = myPCA(dados)

#%%%%%%%%%%%% Kernel PCA
model = KernelPCA(n_components=2, kernel='rbf')   
dados_kpca = model.fit_transform(dados)
dados_kpca = dados_kpca.T

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=20, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Lap. Eig.
model = SpectralEmbedding(n_neighbors=20, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

#%%%%%%%%%%%%% t-SNE
model = TSNE(n_components=2, perplexity=30)
dados_tsne = model.fit_transform(dados)
dados_tsne = dados_tsne.T

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T

#%%%%%%%%%%% Supervised classification
L_pca = Classification(dados_pca, target, 'PCA')
L_kpca = Classification(dados_kpca, target, 'KPCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')
L_tsne = Classification(dados_tsne, target, 't-SNE')
L_umap = Classification(dados_umap, target, 'UMAP')


#%%%%%%%%%%%%%%%%%%% KDE-based ISOMAP
# Number of points for all non-parametric densities
delta = 256     # reduce it to accelerate the computations

# Number of neighbors in KNN graph
ini = 1     # in some datasets the inicial percentile has to be 2
inc = 1
percs = list(range(ini, 21, inc))   # 20 different models based on variations of the e-neighborhood graph
acuracias = []
scs = []

# Model selection: we create 20 different models and choose the one which maximizes accuracy
for p in percs:
    print('Percentil = %d' %p)
    dados_npiso = NonParamISO(dados, p, 2)
    dados_npiso = dados_npiso.T
    L_npiso = Classification(dados_npiso, target, 'NP-ISO')
    scs.append(L_npiso[0])
    acuracias.append(L_npiso[1])

# Plot the accuracies to choose the best model 
print('List of values for percentiles: ', percs)
print('Supervised classification accuracies: ', acuracias)
acuracias = np.array(acuracias)
print('Max Acc: ', acuracias.max())
print('K* = ', percs[acuracias.argmax()])
print()

plt.figure(1)
plt.plot(percs, acuracias)
plt.title('Maximum accuracies for different values of percentiles')
plt.show()

# Plot the SC's (another option to choose the best model)
print('List of values for percentiles: ', percs)
print('Silhouette Coefficients: ', scs)
scs = np.array(scs)
print('Max SC: ', scs.max())
print('K* = ', percs[scs.argmax()])
print()

plt.figure(2)
plt.plot(percs, scs, color='red')
plt.title('Silhouette coefficients for different values of percentiles')
plt.show()