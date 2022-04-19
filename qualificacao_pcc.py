#!/usr/bin/env python3
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
import tensorflow as tf
import random
import time
import umap
import os
from sklearn import decomposition
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from pcc import ParticleCompetitionAndCooperation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


#print(os.listdir("/home/jeffersonpasserini/dados/ProjetosPos/via-dataset/images/"))


os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/via-dataset/images/"

#dimension reduction model
model_dimension_reduction = 'PCA' # PCA ou UMAP

#dimension reduction intitial parameter
number_reduce_components=24

#PCC parameters
perc_samples = 0.1
n_knn_neighbors = 24 #a partir desde numero+1 os testes começam
n_neighbors_test = 10 #number neghbors testing 
v_p_grd = 0.5
v_delta_v=0.1
v_max_iter=1000000

#tests parameters
n_tests = 50


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


#FUNCTION FOR ENCODING STRING LABELS AND GENERATING "UNLABELED DATA"
def hideLabelsKF(labels, test_kf):

    masked_labels = labels.copy()
    
    for position in test_kf:
        masked_labels[position]=-1
        
    return masked_labels


def create_model(model_type):
    
    #CNN Parameters
    IMAGE_CHANNELS=3
    POOLING = None # None, 'avg', 'max'
    
    # load model and preprocessing_function
    if model_type=='VGG16':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input   
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='Xception':
        image_size = (299, 299)
        #from tensorflow.keras.applications.xception import Xception, preprocess_input
        #model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        #from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        #model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        #from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        #model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        #from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        #model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        #from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        #model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        #from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        #model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        #from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        #model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        #from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        #model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        #from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        #model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        #from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        #model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        #from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        #model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        #from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        #model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        #from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        #model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        #from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        #model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        #from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        #model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        #from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        #model = MobileNetV2(weighnumber_neigh_knn = n_knn_neighborsts='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        #model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        #model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        #model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        #model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        #model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        #model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        #model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        #from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        #model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    
    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)
        
    return model, preprocessing_function, image_size

def feature_model_extract(model_type):
    
    start = time.time()
    
    #extracting features
    if model_type=='VGG16':
        print('Extract features VGG16')
        model_type = 'VGG16'
        modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
        features = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
    elif model_type=='VGG19':
        print('Extract features VGG19')
        model_type = 'VGG19'
        modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
        features = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)
    elif model_type=='VGG16+VGG19':
        print('Extract features VGG16+VGG19')
        model_type = 'VGG16'
        modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
        features_VGG16 = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
            
        model_type = 'VGG19'
        modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
        features_VGG19 = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)
        
        #concatenate array features VGG16+VGG19
        features = np.hstack((features_VGG16,features_VGG19))
        
    else: print("Error: Model not implemented.")
    
    end = time.time()
    
    time_feature_extration = end-start
    
    return features, time_feature_extration

def dimensinality_reduction(model_type_reduction, number_components):
    
    start = time.time()
    
    if (model_type_reduction=='PCA'):
        print("dimensionality reduction ..." )
        reduction = decomposition.PCA(n_components=number_components)
        components = reduction.fit_transform(features)
    else:
        print('scaled features')
        scaled_features = StandardScaler().fit_transform(features)
        print('define umap model')
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=number_components, metric='euclidean')
        print('apply umap reducer')
        components = reducer.fit_transform(scaled_features)
    
    end = time.time()
    
    time_reduction = end-start
    
    return components, time_reduction


#----------------------- Main ------------------------------------------------

print("Pre processing...")
model_type_list = ('VGG16', 'VGG19', 'VGG16+VGG19')

#load data
df = load_data()

#lables array
labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)

# creating folds for cross-validation
kfold_n_splits = 10
kfold_n_repeats = 1
kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)
kf.split(df)

model_type = 'VGG16+VGG19'

features, time_feature_extration = feature_model_extract(model_type)

print('time feature extraction --> ')
print(time_feature_extration)
    

if (model_dimension_reduction!=''):
    components, time_reduction = dimensinality_reduction(model_dimension_reduction, number_reduce_components)

    print('time reduction --> ')
    print(time_reduction)
else:
    print("Processing with all features extracted...")
    components = features

number_neigh_knn = n_knn_neighbors

acc_score = []
roc_score = []
f1c_score = []
resultkf = []
time_classifier = []
media = []
std = []

for index, [train, test] in enumerate(kf.split(df)):

    start = time.time()
    print('index: '+str(index))
    
    print("mascara rotulos")
    #mascarar os rotulos para gerar as amostras para o modelo
    masked_labels = hideLabelsKF(labels,test)
        
    #RUN THE MODEL
    print("PCC iniciado...")
    model = ParticleCompetitionAndCooperation()
    model.build_graph(components,k_nn=number_neigh_knn)
    pred = np.array(model.fit_predict(masked_labels, p_grd=v_p_grd, delta_v=v_delta_v, max_iter=v_max_iter))
    end = time.time()
    
    time_pcc = end-start
    
    print('time PCC prediction --> ')
    print(time_pcc)
    
    #SEPARATE PREDICTED SAMPLES
    hidden_labels = np.array(labels[masked_labels == -1]).astype(int)
    hidden_pred = pred[masked_labels == -1]
    
    #PRINT ACCURACY SCORE
    print("Comp:" + str(number_reduce_components) + " -knn:" + str(number_neigh_knn) + " Exec:" + str(index) + " - Acc Score:" + "{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)) + " f1 Score:" + "{0:.4f}".format(f1_score(hidden_labels,hidden_pred)) + " ROC Score:" + "{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)) + " Execution Time: " + "{0:.4f}".format(end-start) +'s')
    acc_score.append(accuracy_score(hidden_labels,hidden_pred))
    roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
    f1c_score.append(f1_score(hidden_labels,hidden_pred))
    time_classifier.append((end-start))
    resultkf.append([number_reduce_components, number_neigh_knn, accuracy_score(hidden_labels,hidden_pred), f1_score(hidden_labels,hidden_pred), roc_auc_score(hidden_labels,hidden_pred), (end-start)])    

print('--------------------------------------------------------------------------------------------------')
media.append([number_reduce_components, number_neigh_knn, np.mean(acc_score), np.mean(f1c_score), np.mean(roc_score), np.mean(time_classifier)])
std.append([number_reduce_components, number_neigh_knn, np.std(acc_score), np.std(f1c_score), np.std(roc_score), np.std(time_classifier)])    




