# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:28:00 2021

@author: Jefferson Passerini / Fabricio Breve

1 - Extract features from the input dataset using VGG16 and VGG19

2 - Dimensionality reduction array features VGG16+VGG19 -PCA/UMAP

3 - Classification images PCC
"""


import numpy as np
import pandas as pd
import random
import time
import umap
import os
from sklearn import decomposition
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from pcc import ParticleCompetitionAndCooperation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.preprocessing import normalize

#print(os.listdir("../via-dataset/images/"))


#CNN Parameters
IMAGE_CHANNELS=3
POOLING = None # None, 'avg', 'max'

DATASET_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/via-dataset/images/"

#define modelo de red. dimensionalidade
algDim = 'UMAP' # PCA ou UMAP

#pca parameters inicial
#number_pca_components=1
number_reduce_components=22

#PCC parameters
perc_samples = 0.1
n_knn_neighbors = 5 #a partir desde numero+1 os testes começam
n_neighbors_test = 5 #number neghbors testing 
v_p_grd = 0.5
v_delta_v=0.1
v_max_iter=1000000

#tests parameters
n_tests = 1

def load_data():
    
    filenames = os.listdir(DATASET_PATH)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(1)
        else:
            categories.append(0)
    
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def create_model(model_type):
    # load model and preprocessing_function
    image_size = (224, 224)
    if model_type=='VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    else: print("Error: Model not implemented.")
    
    preprocessing_function = preprocess_input
    
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model

    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function, image_size

def extract_features(df, model, preprocessing_function, image_size):
    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'}) 
           
    datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    
    total = df.shape[0]
    batch_size = 4
    
    generator = datagen.flow_from_dataframe(
        df, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator, steps=np.ceil(total/batch_size))
    
    return features
   
#FUNCTION FOR ENCODING STRING LABELS AND GENERATING "UNLABELED DATA"
def hideLabels(true_labels, percentage):
    
    mask = np.ones((1,len(true_labels)),dtype=bool)[0]
    
    labels = true_labels.copy()
    
    for l, enc in zip(np.unique(true_labels),range(0,len(np.unique(true_labels)))):
        
        deck = np.argwhere(true_labels == l).flatten()
        random.shuffle(deck)
        
        mask[deck[:round(percentage * len(true_labels[true_labels == l]))]] = False

        labels[labels == l] = enc

    labels[mask] = -1
    
    return np.array(labels).astype(int)


#----------------------- Main ------------------------------------------------
    
model_type_list = ('VGG16', 'VGG19')
df = load_data()

#extracting data from VGG16 and VGG19 CNN   
print("Extracting features - VGG16 ..." )
model_type = 'VGG16'
modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
features_VGG16 = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
    
print("Extracting features - VGG19 ..." )
model_type = 'VGG19'
modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
features_VGG19 = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)

#concatenate array features VGG16+VGG19
print("concatented features array - VGG16+VGG19 ..." )
features = np.hstack((features_VGG16,features_VGG19))
                     
#gerar array de rotulos
labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)

#inicializa resultados
resultado = []
media = []
std = []

acc_score = []
roc_score = []
f1c_score = []
tempo = []

number_components = number_reduce_components
for x in range(0, 5):
    #incrementa o numero de vizinhos proprios
    number_components+=1
    number_neigh_knn = n_knn_neighbors
    
    #Dimensionalaty reduction VGG16+VGG19 features array
    if (algDim=='PCA'):
        print("dimensionality reduction ..." )
        reduction = decomposition.PCA(n_components=number_components)
        pca_components = reduction.fit_transform(features)
    elif (algDim=='UMAP'):
        print('scaled features')
        scaled_features = StandardScaler().fit_transform(features)
        print('define umap model')
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=number_components, metric='euclidean')
        print('apply umap reducer')
        pca_components = reducer.fit_transform(scaled_features)
    else:
        pca_components = features
    
    for j in range(0, n_neighbors_test):        
        #incrementa numero de vizinhos
        number_neigh_knn+=1
        
        acc_score.clear()
        roc_score.clear()
        f1c_score.clear()
        tempo.clear()
        
        #executa n vezes o modelo com as caracteristicas produzidas
        for i in range(0, n_tests):
        
            #print("mascara rotulos")
            #mascarar os rotulos para gerar as amostras para o modelo
            masked_labels = hideLabels(labels,perc_samples)
        
            #RUN THE MODEL
            #print("PCC iniciado...")
            start = time.time()
            model = ParticleCompetitionAndCooperation()
            model.build_graph(pca_components,k_nn=number_neigh_knn)
            #pred = np.array(model.fit_predict(masked_labels))
            pred = np.array(model.fit_predict(masked_labels, p_grd=v_p_grd, delta_v=v_delta_v, max_iter=v_max_iter))
            end = time.time()
        
            #SEPARATE PREDICTED SAMPLES
            hidden_labels = np.array(labels[masked_labels == -1]).astype(int)
            hidden_pred = pred[masked_labels == -1]
        
            #PRINT ACCURACY SCORE
            print("Comp:" + str(number_components) + " -knn:" + str(number_neigh_knn) + " Exec:" + str(i+1) + " - Acc Score:" + "{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)) + " f1 Score:" + "{0:.4f}".format(f1_score(hidden_labels,hidden_pred)) + " ROC Score:" + "{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)) + " Execution Time: " + "{0:.4f}".format(end-start) +'s')
            acc_score.append(accuracy_score(hidden_labels,hidden_pred))
            roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
            f1c_score.append(f1_score(hidden_labels,hidden_pred))
            tempo.append((end-start))
            resultado.append([number_neigh_knn, number_components, accuracy_score(hidden_labels,hidden_pred), f1_score(hidden_labels,hidden_pred), roc_auc_score(hidden_labels,hidden_pred), (end-start)])    
        
        print('--------------------------------------------------------------------------------------------------')
        media.append([number_components, number_neigh_knn, np.mean(acc_score), np.mean(f1c_score), np.mean(roc_score), np.mean(tempo)])
        std.append([number_components, number_neigh_knn, np.std(acc_score), np.std(f1c_score), np.std(roc_score), np.std(tempo)])    
        
        

np.savetxt('resultado_'+algDim+'.csv', resultado, fmt='%.4f', delimiter=';')
np.savetxt('resultado_'+algDim+'_avg.csv', media, fmt='%.4f', delimiter=';')
np.savetxt('resultado_'+algDim+'_std.csv', std, fmt='%.4f', delimiter=';')

#PRINT CONFUSION MATRIX
#print("Confusion Matrix:\n", confusion_matrix(hidden_labels, hidden_pred))
#print("Confusion Matrix:\n", classification_report(hidden_labels, hidden_pred))





