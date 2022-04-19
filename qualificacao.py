#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:28:00 2021

@author: Jefferson Passerini / Fabricio Breve

1 - Extract features from the input dataset using CNN

2 - Dimensionality reduction array features PCA

3 - Classification with n classifiers
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
   
#FUNCTION FOR ENCODING STRING LABELS AND GENERATING "UNLABELED DATA FOR PCC Classifier"
def hideLabels(labels, test_kf):

    masked_labels = labels.copy()
    
    for position in test_kf:
        masked_labels[position]=-1
        
    return masked_labels


def gen_dataset(features, labels, train, test):
    
    dataset_train = np.array(features[train])
    dataset_train_label = np.array(labels[train])
    
    dataset_test = np.array(features[test])
    dataset_test_label = np.array(labels[test])
    
    return dataset_train, dataset_train_label, dataset_test, dataset_test_label


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
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                
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
    if model_type=='VGG16+VGG19':
        print('Extract features '+model_type)       
        model_type = 'VGG16'
        modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
        features_VGG16 = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
            
        model_type = 'VGG19'
        modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
        features_VGG19 = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)
        
        #concatenate array features VGG16+VGG19
        features = np.hstack((features_VGG16,features_VGG19))
        
    elif model_type=='Xception+ResNet50':
        print('Extract features '+model_type)       
        model_type = 'Xception'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'ResNet50'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))

    else: 
        print('Extract features '+model_type)       
        model, preprocessing_function, image_size = create_model(model_type)
        features = extract_features(df, model, preprocessing_function, image_size)
        
    
    end = time.time()
    
    time_feature_extration = end-start
    
    return features, time_feature_extration

def dimensinality_reduction(model_type_reduction, number_components, allfeatures, stdScaler):
    
    if (stdScaler == 'Yes'):
        allfeatures_Reduction = StandardScaler().fit_transform(allfeatures)
    else:
        allfeatures_Reduction = allfeatures
    
    start = time.time()
    
    if (model_type_reduction=='PCA'):
        print("dimensionality reduction: "+model_type_reduction)
        reduction = decomposition.PCA(n_components=number_components)
        components = reduction.fit_transform(allfeatures_Reduction)
        
    elif (model_type_reduction=='UMAP'):
        print("dimensionality reduction: "+model_type_reduction)
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=number_components, metric='euclidean')
        components = reducer.fit_transform(allfeatures_Reduction)
    
    elif (model_type_reduction=='RELIEFF'):
        print("Error: Model not implemented.")
        
    elif (model_type_reduction=='None'):
        print("Processing with all features extracted...")
        components = allfeatures_Reduction
              
    else: print("Error: Model not implemented.")
        
    
    end = time.time()
    
    time_reduction = end-start
    
    return components, time_reduction

def classification(train_data, train_label, test_data, model_classifier):
    
    if (model_classifier=='PCC'):
        print("Classification - "+model_classifier)
        
        time_trainning = 0
        start = time.time()
        model = ParticleCompetitionAndCooperation()
        model.build_graph(components,k_nn=n_knn_neighbors)
        classification_result = np.array(model.fit_predict(train_data, p_grd=v_p_grd, delta_v=v_delta_v, max_iter=v_max_iter))
        end = time.time()
        time_prediction = end-start
    
    elif (model_classifier=='J48'):
        print("Classification - "+model_classifier)
        
        
    else: print("Error: Model not implemented.")
        
    return time_trainning, time_prediction, classification_result
    
    

#----------------------- Main ------------------------------------------------
model_type_list = ['Xception+ResNet50','VGG16+VGG19', 'Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 
        'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2', "InceptionV3",
        'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169',
        'DenseNet201', 'NASNetMobile', 'MobileNetV2',
        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
        'EfficientNetB6', 'EfficientNetB7']


#model_reduction_dim_list = ['PCA', 'UMAP', 'ReliefF', 'mRMR','None'] #mRMR Minimum redundancy feature selection
model_reduction_dim_list = ['PCA', 'UMAP'] #mRMR Minimum redundancy feature selection
number_reduce_components=24
scaled_feat_reduction = 'No' # Yes or No

#model_classifier_list = ['PCC', 'J48', 'SMO', 'MLP', 'Logistic', 'RBF']
model_classifier_list = ['PCC']


#PCC parameters
perc_samples = 0.1
n_knn_neighbors = 24 
v_p_grd = 0.5
v_delta_v=0.1
v_max_iter=1000000

# create filenames
data_filename = "data_detailed.csv"
data_acc_filename = "acc_resume.csv"
data_f1_filename = "f1_resume.csv"
data_roc_filename = "roc_resume.csv"
data_time_filename = "time_resume.csv"

#load data
df = load_data()

#lables array
labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)

# creating folds for cross-validation - 10fold
kfold_n_splits = 10
kfold_n_repeats = 1
kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)
kf.split(df)


#CNN loop
for model_type in model_type_list:
    
    print("Pre processing...")
    features, time_feature_extration = feature_model_extract(model_type)
    print('time feature extraction --> '+str(time_feature_extration))

    #reduction loop
    for model_dimension_reduction in model_reduction_dim_list:
        
        components, time_reduction = dimensinality_reduction(model_dimension_reduction, number_reduce_components,features,scaled_feat_reduction)
        print('time reduction --> '+str(time_reduction))
        
        #classifier loop
        for model_classifier in model_classifier_list:
    
            #log array's
            acc_score = []
            roc_score = []
            f1c_score = []
            data_time_trainning = []
            data_time_prediction = []
        
            print(model_type + " >> " + model_dimension_reduction + " >> " + model_classifier + "------------------")
            #kfold loop
            for index, [train, test] in enumerate(kf.split(df)):
            
                dataset_train = [] 
                dataset_train_label = []
                dataset_test = [] 
                dataset_test_label = []
                
                #run classification
                #mascarar os rotulos para gerar as amostras para o modelo
                if(model_classifier == 'PCC'):
                    masked_labels = hideLabels(labels,test)
                    time_trainning, time_prediction, pred = classification(masked_labels, dataset_train_label, dataset_test, model_classifier)
                    
                else:
                    dataset_train, dataset_train_label, dataset_test, dataset_test_label = gen_dataset(features, train, test)
                    time_trainning, time_prediction, pred = classification(dataset_train, dataset_train_label, dataset_test, model_classifier)
                
                print('index: '+str(index) + 'Trainning --> '+str(time_trainning) + 'Prediction --> '+str(time_prediction))
                print()
                
                #SEPARATE PREDICTED SAMPLES
                if (model_classifier == 'PCC'):
                    hidden_labels = np.array(labels[masked_labels == -1]).astype(int)
                    hidden_pred = pred[masked_labels == -1]
                else:
                    hidden_labels = dataset_test_label.copy()
                    hidden_pred = pred.copy()
                
                #csv detailed data
                with open(data_filename,"a+") as f_data:
                    f_data.write(model_type+", ") #CNN
                    f_data.write(model_dimension_reduction+", ") #Reduction_alg
                    f_data.write(model_classifier+", ") #Classifier
                    f_data.write(str(index+1)+", ") #Kfold index
                    f_data.write( "'"+str(np.shape(features))+"', " ) #CNN_features
                    f_data.write(scaled_feat_reduction+", ") #Reduction_Scaled
                    f_data.write( "'"+str(np.shape(components))+"', " ) #Reduction_Components
                    f_data.write(str(n_knn_neighbors)+", ")  #k_neigh_PCC_classifier
                    f_data.write(str(accuracy_score(hidden_labels,hidden_pred)*100)+", ") #Acc Score
                    f_data.write(str(f1_score(hidden_labels,hidden_pred)*100)+", ") #F1 Score
                    f_data.write(str(roc_auc_score(hidden_labels,hidden_pred)*100)+", ") #ROC Score
                    f_data.write(str(time_feature_extration)+", ") #Time Extraction Features
                    f_data.write(str(time_reduction)+", ") #Time Reduction dimensionality
                    f_data.write(str(time_trainning)+", ") #Time Classifier Trainning
                    f_data.write(str(time_prediction)+", \n") #Time Classifier Predict
                    
                #PRINT ACCURACY SCORE
                print("Comp:" + str(number_reduce_components) + " -knn:" + str(n_knn_neighbors) + " Exec:" + str(index) + " - Acc Score:" + "{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)) + " f1 Score:" + "{0:.4f}".format(f1_score(hidden_labels,hidden_pred)) + " ROC Score:" + "{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)) + " Execution Time: " + "{0:.4f}".format(time_prediction+time_trainning) +'s')
                
                #score's log
                data_time_prediction.append(time_prediction)
                data_time_trainning.append(time_trainning)
                acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                print("-------------------------------------------------------------------------------------------------------")

#log acc 
with open(data_acc_filename,"a+") as f_acc_csv:
    f_acc_csv.write(model_type+", ") #CNN
    f_acc_csv.write(model_dimension_reduction+", ") #Reduction_alg
    f_acc_csv.write(model_classifier+", ") #Classifier
    for acc in acc_score:
        f_acc_csv.write(str("%.4f" % acc)+", ")
    f_acc_csv.write("\n")
    
#log f1 score
with open(data_f1_filename,"a+") as f_f1_csv:
    f_f1_csv.write(model_type+", ") #CNN
    f_f1_csv.write(model_dimension_reduction+", ") #Reduction_alg
    f_f1_csv.write(model_classifier+", ") #Classifier
    for f1sc in f1c_score:
        f_f1_csv.write(str("%.4f" % f1sc)+", ")
    f_f1_csv.write("\n")
    
#log roc score
with open(data_roc_filename,"a+") as f_roc_csv:
    f_roc_csv.write(model_type+", ") #CNN
    f_roc_csv.write(model_dimension_reduction+", ") #Reduction_alg
    f_roc_csv.write(model_classifier+", ") #Classifier
    for roc_sc in roc_score:
        f_roc_csv.write(str("%.4f" % roc_sc)+", ")
    f_roc_csv.write("\n")



#Result (Detailed Data)
#with open(data_filename,"a+") as f_data:
#    f_data.write("cnn model, ") #CNN
#    f_data.write("reduction alg, ") #Reduction_alg
#    f_data.write("classifier, ") #Classifier
#    f_data.write("Kfold, ") #Kfold index
#    f_data.write("features, " ) #CNN_features
#    f_data.write("scaled, ") #Reduction_Scaled
#    f_data.write("components, " ) #Reduction_Components
#    f_data.write("PCC neighs, ")  #k_neigh_PCC_classifier
#    f_data.write("Acc, ") #Acc Score
#    f_data.write("F1, ") #F1 Score
#    f_data.write("Roc, ") #ROC Score
#    f_data.write("Extraction time, ") #Time Extraction Features
#    f_data.write("Reduction time, ") #Time Reduction dimensionality
#    f_data.write("Classifier Trainning, ") #Time Classifier Trainning
#    f_data.write("Classifier Predict, \n") #Time Classifier Predict


#Resume Data ACC
#CNN
#Reduction_alg
#Classifier
#CNN_features
#Reduction_Scaled
#Reduction_Components
#k_neigh_PCC_classifier
#1fold_acc
#2fold_acc
#3fold_acc
#4fold_acc
#5fold_acc
#6fold_acc
#7fold_acc
#8fold_acc
#9fold_acc
#10fold_acc



