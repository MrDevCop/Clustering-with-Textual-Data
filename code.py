#conda env create -f environemnts-gpu.yml
#source activate behavioural-cloning# 
from newsapi import NewsApiClient
import requests
import spacy
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AffinityPropagation
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sp = spacy.load('en_core_web_lg')

#%%
#-----------------------------Creating dataset with the following keywords------------------------
def News():
    
    
    keywords = ['economy', 'royal family', 'bitcoin', 'AI', 'music', 'art', 'lockdown', 'pakistan', 'football', 'syria', 'politics', 'immigration', 'terrorism', 'tourism', 'imran khan', 'elections', 'entertainment', 'pfizer', 'vaccine', 'recession', 'airlines', 'myanmar', 'martial law', 'democracy', 'corruption']
    API_KEY = '38ac93ffa0d0416a8a2d024ae890f1fb'

    if os.path.exists("dataset.txt"):
        print("Dataset file already exists")
    
    else:    
    
        for j in range(len(keywords)):    

            params = {
                    'q': keywords[j],
                    #'source': 'bbc-news',
                    #'sortBy': 'All',
                    'language': 'en',
                    #'category': 'business',
                    #'country': 'us',
                    #'apiKey': API_KEY,
                    }

            headers = {'X-Api-Key': API_KEY,  # KEY in header to hide it from url
                       }

            url = 'https://newsapi.org/v2/everything'

            request = requests.get(url, params=params, headers=headers)
            request_json = request.json()
            article = request_json["articles"]
            news = []
            results = []
    
            for ar in article:
        
                results.append(ar["title"])
    
            print(len(results))
            sep = ' - '
            temp = ''
            stripped = ''
    
            for i in range(len(results)):
        
                temp = results[i]
                stripped = temp.split(sep,1)[0] 
                news.append(stripped)
        
            file = open (r"dataset.txt","a", encoding = 'utf-8')  
            news=map(lambda x:x+'\n', news)
            file.writelines(news) 
            file.close()

#%%
News()

#%%
#-------------------------------Removing stop words, lemmatization--------------------------
# Reading from the file
def sw_tok_lemm(filename):
    
    data = []
    with open(filename,'r',encoding="utf8") as f:
        data = f.readlines()
    f.close()

    documents = []

# Removing stop words, tokenizing, and lemmatization
    stopwords = sp.Defaults.stop_words
    for d in data:
        d_nlp = sp(d.lower())
        t_list = []
        for token in d_nlp:
            tok_lem = str(token.lemma_)
            if (tok_lem not in stopwords):
                t_list.append(tok_lem)
        str_ = ' '.join(t_list) 
        documents.append(str_)
    return (documents)


#%%
#------------------Finds best number of clusters using scree plot----------------------------

def scree_plot(matrix):
    sse = []
    for k in range(2, 16):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(matrix)
        sse.append(kmeans.inertia_)
    print(sse)
    fig = plt.figure(figsize=(35, 5))
    plt.plot(range(2, 16), sse)
    plt.grid(True)
    plt.title('Scree Plot')

    for i in range(len(sse)):
        print((i+2),sse[i])

    k = np.argmin(sse) + 2
    
    return (k)

#%%
#----------------------finding the best numbers of clusters using dendogram-------------------------
# This function is for binary and tf methods only
    
def dendogram_(matrix):
    
    linked = linkage(matrix, 'ward')

    labelList = range(1, matrix.shape[0]+1)

# drawing dendogram

    plt.figure(figsize=(20, 7))
    dendrogram(linked, orientation='top',labels=labelList,distance_sort='descending',show_leaf_counts=True)
    plt.title("Dendogram")
    cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', compute_full_tree=True, distance_threshold=8)
    cluster.fit_predict(matrix)
    k =1+np.amax(cluster.labels_)
    plt.axhline(y=8, color='r', linestyle='--')
    plt.show()
    print('\nThe number of clusters as determined by dendogram is: ', k)
    print("\n")
    return(k)
    
#%%
#--------------------------finds the best number of clusters using dendogram-------------- 
# dendogram function for tf-idf matrix

def dendogram_tfidf(matrix):
    
    linked = linkage(matrix, 'ward')

    labelList = range(1, matrix.shape[0]+1)

# drawing dendogram

    plt.figure(figsize=(20, 7))
    dendrogram(linked, orientation='top',labels=labelList,distance_sort='descending',show_leaf_counts=True)
    plt.title("Dendogram")
    cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', compute_full_tree=True, distance_threshold=3)
    cluster.fit_predict(matrix)
    k =1+np.amax(cluster.labels_)
    plt.axhline(y=30, color='r', linestyle='--')
    plt.show()
    print('\nThe number of clusters as determined by dendogram is: ', k)
    print("\n")
    return(k)
    
#%%
#---------------------finding the best value of clusters using silhoutte coefficient-----------

def silhouette_(matrix):
    
    s_avg = []
    for k in range(2, 16):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(bin_matrix)
        sil_sc = silhouette_score(bin_matrix, kmeans.labels_)
        s_avg.append(sil_sc)
    print(s_avg)

    plt.figure(figsize=(20,7))
    plt.title("Avg Silhoutte")
    ax = plt.axes()
    x=s_avg
    y=range(2,16)

    ax.plot(y,x)
    xx=str(max(s_avg))
    
    k  = np.argmax(s_avg)+2
    print("\nMaximum avg silhoutte coefficent is: " + xx + " which corresponds to k = ", k)
    print("\n")
    return (k)
#%%
#--------finding the best number of clusters using afinity propagation---------------------

def affinity_prop(matrix):
    
    aff_prop= AffinityPropagation(max_iter = 5000, verbose = True, preference = -23)
    aff_prop.fit(bin_matrix)
    
    cluster_centers_indices = aff_prop.cluster_centers_indices_
    

    k = len(cluster_centers_indices)
    return (k)


#%%
#-----------------performs clustering algorithm and outputs the best cluster------------
def cluster_(matrix, k_scree, k_dendo, k_sil, k_aff_prop):
    
    km1 = KMeans(n_clusters =k_scree, init = 'k-means++', n_init = 70, algorithm ="elkan", max_iter = 10000, random_state = 6)
    km1.fit(matrix)

    km2 = KMeans(n_clusters =k_dendo, init = 'k-means++', n_init = 70, algorithm ="elkan", max_iter = 10000, random_state = 9)
    km2.fit(matrix)

    km3 = KMeans(n_clusters =k_sil, init = 'k-means++', n_init = 70, algorithm ="elkan", max_iter = 10000, random_state = 6)
    km3.fit(matrix)

    km4 = KMeans(n_clusters =k_aff_prop, init = 'k-means++', n_init = 70, algorithm ="elkan", max_iter = 10000, random_state = 6)
    km4.fit(matrix)
    k = [k_scree, k_dendo, k_sil, k_aff_prop]

    s1=silhouette_score(matrix,km1.labels_)
    s2=silhouette_score(matrix,km2.labels_)
    s3=silhouette_score(matrix,km3.labels_)
    s4=silhouette_score(matrix,km4.labels_)
    s = [s1,s2,s3,s4]

    d = dict(zip(k, s))
    print(d)
    max_sil_ind = np.argmax(s)
    max_sil = max(s)
    ideal_k = k[max_sil_ind]
    print("\nThe maximum silhouette coefficient of " + str(max_sil) +" proves that the ideal number of clusters is ", ideal_k)
    print("\n")
    return (ideal_k,max_sil)

#%%
#-------------------finds the best number of n_components, k or rank--------------------
# For that we will retain those ranks which successfully explains the 70% variance of the data
    
def ideal_k(matrix):
    
# Assuming n_components = 300 initially
    svd_model = TruncatedSVD(n_components=300, algorithm='randomized', n_iter=1000, random_state=5)
    lsa = svd_model.fit_transform(matrix)
    non_cumulative_sum=svd_model.explained_variance_ratio_
    print ("\nProportion of Variance Explained : ", non_cumulative_sum)  
    out_sum = np.cumsum(svd_model.explained_variance_ratio_)  
    print ("\nTotal Variance Explained: ", out_sum)
    len(svd_model.explained_variance_ratio_)

    i =0
    while out_sum[i]<=0.70:
        i=i+1
        
    k = i
    threshold_value = str(out_sum[k])
    print("\nThreshold value of "+ threshold_value + " is reached at n_components = ",k)
    print("\nHence 70% of variance of the data is explained when k or rank = ",k) 
    print("\n")
    return(k)
    
#%%
#-----------topic modelling by using the ideal number of k found by the previous function-----------------
def topic_modelling(matrix, n_comp, count_vec):
    
    svd_matrix = TruncatedSVD(n_components=n_comp, algorithm='randomized', n_iter=1000, random_state = 6)
    lsa_matrix = svd_matrix.fit_transform(matrix)
    dictionary_matrix = count_vec.get_feature_names()
    lsa_matrix.shape
    svd_matrix.components_.shape

    svd_df_matrix = pd.DataFrame(svd_matrix.components_.T)
    dict_list_matrix = np.array(dictionary_matrix).T
    svd_df_matrix['terms'] = dict_list_matrix
    tail = svd_df_matrix.tail(20)
    print(tail)

#%%
#------------------------------------Final Results---------------------------------------
def results():
    
    n_cluster_list = [n_cluster_bin , n_cluster_tf , n_cluster_tfidf] # best number of clusters acheived by each  pre-processing method
    max_silhoutte_list = [max_silhouette_bin , max_silhouette_tf , max_silhouette_tfidf] # best silhoutte coeff acheived by each pre processing method
    methods = ["Binary-Matrix", "TF-Matrix", "TFIDF-Matrix"]
    dic= dict(zip(methods,max_silhoutte_list))
    indx = np.argmax(max_silhoutte_list)
    max_sil_overall = max(max_silhoutte_list)
    ideal_cluster_overall = n_cluster_list[indx]


    best_method = max(dic, key = dic.get)
    print("\nThe list shows each pre-processing method against its acheived maximum silhouette coefficient after clustering" , dic)
    print("\n")

    print("\nThe best overall pre-processing method is:  "+ str(best_method) + " which gives the maximum silhouette coefficient of", max_sil_overall )
    print("\nThe ideal number of clusters obtained with "+ str(best_method) + " is: ", ideal_cluster_overall)
    print("\n")

#%%
    
documents= sw_tok_lemm('dataset.txt')

#%%
#-------------------------------------------------------------------------------------------
#-------------------------This block of code does the following:-------------------------------------
# Makes a binary-freq matrix
# Makes a scree plot using binary0freq matrix to determine number of clusters
# Makes a dendogram using binary-freq matrix to determine number of clusters
# Makes a silhoutte curve using binary-freq matrix to determine number of clusters
# Uses affinity propagation on binary-freq matrix to determine number of clusters
# Applies clustering algorithm using the number of k found via scree, dendo, silhouette and ap methods
# Determines the best output of clustering acheived among different clusterings
# Determines the ideal value of k or rank
# uses the ideal value of k found in previous step for topic modelling
#---------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

# Every function is called on binary-freq matrix
# making binary-freq matrix
count_vectorizer_binary = CountVectorizer(documents, binary = True)
bin_matrix= count_vectorizer_binary.fit_transform(documents).toarray()

# k value obtained from scree plot
k_scree_bin = scree_plot(bin_matrix)
print('\nThere is no definite elbow in the plot yet the number of clusters as determined by Scree plot and the minimum SSE is :' , k_scree_bin)

# k value obtained from dendogram
k_dendogram_bin = dendogram_(bin_matrix)

# k value obtained from silhouette curve
k_sil_bin = silhouette_(bin_matrix)
print('\nThe number of clusters as determined by silhouette curve is: ', k_sil_bin)

# k value obtained from affinity propagation
k_aff_prop_bin = affinity_prop(bin_matrix)
print('\nThe number of clusters as determined by affinity propagation algorithm is: ', k_aff_prop_bin)

# Results of clustering algorithm
n_cluster_bin,max_silhouette_bin = cluster_(bin_matrix, k_scree_bin, k_dendogram_bin, k_sil_bin, k_aff_prop_bin)

# Ideal value of rank or k
ideal_n_comp_bin = ideal_k(bin_matrix)

# topic modelling
topic_modelling(bin_matrix, ideal_n_comp_bin, count_vectorizer_binary)

#%%
#------------------------------------------------------------------------------------------
#-------------------------This block of code does the following:-------------------------------------
# Makes a tf matrix
# Makes a scree plot using tf matrix to determine number of clusters
# Makes a dendogram using tf matrix to determine number of clusters
# Makes a silhoutte curve using tf matrix to determine number of clusters
# Uses affinity propagation on tf matrix to determine number of clusters
# Applies clustering algorithm using the number of k found via scree, dendo, silhouette and ap methods
# Determines the best output of clustering acheived among different clusterings
# Determines the ideal value of k or rank
# uses the ideal value of k found in previous step for topic modelling
#---------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# Every function is called on tf matrix
# making term-frequency matrix
count_vectorizer_tf = CountVectorizer(documents, binary = False)
tf_matrix = count_vectorizer_tf.fit_transform(documents).toarray()

# k value obtained from scree plot
k_scree_tf = scree_plot(tf_matrix)
print('\nThere is no definite elbow in the plot yet the number of clusters as determined by Scree plot and the minimum SSE is :' , k_scree_tf)

# k value obtained from dendogram
k_dendogram_tf = dendogram_(tf_matrix)

# k value obtained avg silhouette curve
k_sil_tf = silhouette_(tf_matrix)
print('\nThe number of clusters as determined by silhouette curve is: ', k_sil_tf)

# k value obtained from affinity propagation
k_aff_prop_tf = affinity_prop(tf_matrix)
print('\nThe number of clusters as determined by affinity propagation algorithm is: ', k_aff_prop_tf)

# Results of clustering algorithm
n_cluster_tf ,max_silhouette_tf = cluster_(tf_matrix, k_scree_tf,k_dendogram_tf, k_sil_tf, k_aff_prop_tf)

# Ideal value of rank or k
ideal_n_comp_tf = ideal_k(tf_matrix)
print('\nThe ideal value of k or rank is: ', ideal_n_comp_tf)

# topic modelling
topic_modelling(tf_matrix, ideal_n_comp_tf, count_vectorizer_tf)
    

#%%
#-----------------------------------------------------------------------------------------
#-------------------------This block of code does the following:-------------------------------------
# Makes a tfidf matrix
# Makes a scree plot using tfidf matrix to determine number of clusters
# Makes a dendogram using tfidf matrix to determine number of clusters
# Makes a silhoutte curve using tfidf matrix to determine number of clusters
# Uses affinity propagation on tfidf matrix to determine number of clusters
# Applies clustering algorithm using the number of k found via scree, dendo, silhouette and ap methods
# Determines the best output of clustering acheived among different clusterings
# Determines the ideal value of k or rank
# uses the ideal value of k found in previous step for topic modelling
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# Every function is called on tfidf matrix
# making tf-idf matrix
count_vectorizer_tfidf = TfidfVectorizer(documents)
tfidf_matrix = count_vectorizer_tfidf.fit_transform(documents).toarray()

# k value obtained from scree plot
k_scree_tfidf = scree_plot(tfidf_matrix)
print('\nThere is no definite elbow in the plot yet the number of clusters as determined by Scree plot and the minimum SSE is :' , k_scree_tfidf)

# k value obtained from dendogram
k_dendogram_tfidf = dendogram_tfidf(tfidf_matrix)

# k value obtained from silhouette curve
k_sil_tfidf = silhouette_(tfidf_matrix)
print('\nThe number of clusters as determined by silhouette curve is: ', k_sil_tfidf)

# k value obtained from affinity propagation
k_aff_prop_tfidf = affinity_prop(tfidf_matrix)
print('\nThe number of clusters as determined by affinity propagation algorithm is: ', k_aff_prop_tfidf)

# Results of clustering algorithm
n_cluster_tfidf, max_silhouette_tfidf = cluster_(tfidf_matrix, k_scree_tfidf, k_dendogram_tfidf, k_sil_tfidf, k_aff_prop_tfidf)

# Ideal value of rank or k
ideal_n_comp_tfidf = ideal_k(tfidf_matrix)

# topic modelling
topic_modelling(tfidf_matrix, ideal_n_comp_tfidf, count_vectorizer_tfidf)

#%%
results()


