#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:28:00 2021

@author: Jefferson Passerini / Fabricio Breve

1 - Extract features from the input dataset using CNNs

2 - Dimensionality reduction array features PCA/UMAP or 
    Feature Selection with ReliefF / mRMR

3 - Classification with n classifiers
"""
#base libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os
#transformation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
#dimensionality reduction
from sklearn import decomposition
import umap
#feature selection
from ReliefF import ReliefF
#classifier's
from pcc import ParticleCompetitionAndCooperation
from sklearn.tree import DecisionTreeClassifier #J48
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#metrics
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
RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/"

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
        model_type = 'VGG16'
        modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
        features_VGG16 = extract_features(df, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
            
        model_type = 'VGG19'
        modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
        features_VGG19 = extract_features(df, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)
        
        #concatenate array features VGG16+VGG19
        features = np.hstack((features_VGG16,features_VGG19))
        
    elif model_type=='Xception+ResNet50':
        model_type = 'Xception'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'ResNet50'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))

    else: 
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
        reduction = decomposition.PCA(n_components=number_components)
        components = reduction.fit_transform(allfeatures_Reduction)
        
    elif (model_type_reduction=='UMAP'):
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=number_components, metric='euclidean')
        components = reducer.fit_transform(allfeatures_Reduction)
        
    elif (model_type_reduction=='None'):
        print("Processing with all features extracted... \n")
        components = allfeatures_Reduction
              
    else: print("Error: Model not implemented. \n")
        
    end = time.time()
    time_reduction = end-start
    
    return components, time_reduction

def feature_selection(X_train, y_train, X_test, model_type_selection, number_components):
    
    start = time.time()
    
    if (model_type_selection=='ReliefF'):
        #trainning feature selection
         reducer = ReliefF(n_features_to_keep=number_components)
         components_train = reducer.fit_transform(X_train,y_train)
         components_test = reducer.transform(X_test)
         
    else: print("Error: Model not implemented. \n")
        
    end = time.time()
    time_reduction = end-start
    
    return components_train, components_test, time_reduction
    

def classification_pcc(feat_pcc,labels_masked,n_neighbors):
    time_trainning = 0
    start = time.time()
    model = ParticleCompetitionAndCooperation()
    model.build_graph(feat_pcc,k_nn=n_neighbors)
    classification_result = np.array(model.fit_predict(labels_masked, p_grd=v_p_grd, delta_v=v_delta_v, max_iter=v_max_iter))
    end = time.time()
    time_prediction = end-start
    
    return time_trainning, time_prediction, classification_result
    

def classification(train_data, train_label, test_data, model_classifier):
    
    if (model_classifier=='J48'):
        start = time.time()
        clf = DecisionTreeClassifier()        
        clf = clf.fit(train_data,train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
    
    elif (model_classifier=='RBF'):
        start = time.time()
        clf = SVC(kernel='rbf')
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
        
    elif (model_classifier=='LinearSVM'):
        start = time.time()
        clf = SVC(kernel="linear", C=0.025)
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
        
    elif (model_classifier=='MLP'):
        start = time.time()
        clf = MLPClassifier(random_state=1, max_iter=1000)
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
    
    elif (model_classifier=='Logistic'):
        start = time.time()
        clf = LogisticRegression(max_iter=500)
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
        
    elif (model_classifier=='RandomForest'):
        start = time.time()
        clf = RandomForestClassifier()
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
        
    elif (model_classifier=='Adaboost'):
        start = time.time()
        clf = AdaBoostClassifier()
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
    
    elif (model_classifier=='Gaussian'):
        start = time.time()
        clf = GaussianNB()
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
        start = time.time()
        classification_result = clf.predict(test_data)        
        end = time.time()
        time_prediction = end-start
        
    else: print("Error: Model not implemented. \n")
        
    return time_trainning, time_prediction, classification_result
    
  

#----------------------- Main ------------------------------------------------
model_type_list = ['Xception+ResNet50','VGG16+VGG19', 'Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 
        'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2', "InceptionV3",
        'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169',
        'DenseNet201', 'NASNetMobile', 'MobileNetV2',
        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
        'EfficientNetB6', 'EfficientNetB7']

model_type_list = [['MobileNet','Resnet101'],['ResNet101','DenseNet169'],['ResNet101','DenseNet121'],
                   ['ResNet101','MobileNetV2'],['EfficientNetB0','MobileNet'],['MobileNet','ResNet50'],
                   ['Xception','ResNet50'],['VGG16','VGG19']]

#model_type_list = ['EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5']

#model_type_list = ['EfficientNetB7']

model_reduction_dim_list = ['ReliefF'] #mRMR Minimum redundancy feature selection
number_reduce_components=150
scaled_feat_reduction = 'No' # Yes or No

#model_classifier_list = ['SMO']
model_classifier_list = ['PCC', 'J48', 'RBF', 'LinearSVM','MLP','Logistic','RandomForest','Adaboost','Gaussian']

#PCC parameters
perc_samples = 0.1
n_knn_neighbors = 24 
v_p_grd = 0.5
v_delta_v=0.1
v_max_iter=1000000

# create filenames
data_filename = RESULT_PATH+"data_detailed.csv"
data_acc_filename = RESULT_PATH+"acc_resume.csv"
data_f1_filename = RESULT_PATH+"f1_resume.csv"
data_roc_filename = RESULT_PATH+"roc_resume.csv"
data_time_filename = RESULT_PATH+"time_resume.csv"

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
    
    
    #model 01
    
    features_model01, time_feature_extration01 = feature_model_extract(model_type[0])
    print('Extract features '+model_type[0]+' time feature extraction --> '+"{0:.4f}".format(time_feature_extration01) + " \n")
    
    number_features = np.shape(features_model01)[1]
    
    #model 02
    features_model02, time_feature_extration02 = feature_model_extract(model_type[1])
    print('Extract features '+model_type[1]+' time feature extraction --> '+"{0:.4f}".format(time_feature_extration02) + " \n")
    
    number_features = number_features+np.shape(features_model02)[1]
    
    time_feature_extration = time_feature_extration01+time_feature_extration02
    
    model_name = model_type[0]+"+"+model_type[1]
    print('Extract features '+model_name+' time feature extraction --> '+"{0:.4f}".format(time_feature_extration) + " \n")
    

    #reduction loop
    for model_dimension_reduction in model_reduction_dim_list:
        
        #allfeat = features.copy()
        time_reduction = 0
        
        #log array's
        PCC_acc_score = []
        PCC_roc_score = []
        PCC_f1c_score = []
        PCC_data_time_reduction = []
        PCC_data_time_trainning = []
        PCC_data_time_prediction = []
        
        J48_acc_score = []
        J48_roc_score = []
        J48_f1c_score = []
        J48_data_time_reduction = []
        J48_data_time_trainning = []
        J48_data_time_prediction = []
        
        RBF_acc_score = []
        RBF_roc_score = []
        RBF_f1c_score = []
        RBF_data_time_reduction = []
        RBF_data_time_trainning = []
        RBF_data_time_prediction = []
        
        LINEAR_acc_score = []
        LINEAR_roc_score = []
        LINEAR_f1c_score = []
        LINEAR_data_time_reduction = []
        LINEAR_data_time_trainning = []
        LINEAR_data_time_prediction = []
        
        MLP_acc_score = []
        MLP_roc_score = []
        MLP_f1c_score = []
        MLP_data_time_reduction = []
        MLP_data_time_trainning = []
        MLP_data_time_prediction = []
        
        LOG_acc_score = []
        LOG_roc_score = []
        LOG_f1c_score = []
        LOG_data_time_reduction = []
        LOG_data_time_trainning = []
        LOG_data_time_prediction = []
             
        FOREST_acc_score = []
        FOREST_roc_score = []
        FOREST_f1c_score = []
        FOREST_data_time_reduction = []
        FOREST_data_time_trainning = []
        FOREST_data_time_prediction = []
        
        ADA_acc_score = []
        ADA_roc_score = []
        ADA_f1c_score = []
        ADA_data_time_reduction = []
        ADA_data_time_trainning = []
        ADA_data_time_prediction = []
        
        GAU_acc_score = []
        GAU_roc_score = []
        GAU_f1c_score = []
        GAU_data_time_reduction = []
        GAU_data_time_trainning = []
        GAU_data_time_prediction = []
        
        
        #kfold loop
        for index, [train, test] in enumerate(kf.split(df)):
                                                                                                                                                              
            #gera dataset's 
            #model 01
            dataset_train_md01, dataset_train_label_md01, dataset_test_md01, dataset_test_label_md01 = gen_dataset(features_model01, labels, train, test) 
            
            #model 02
            dataset_train_md02, dataset_train_label_md02, dataset_test_md02, dataset_test_label_md02 = gen_dataset(features_model02, labels, train, test) 
            dataset_train_label = dataset_train_label_md01
            dataset_test_label = dataset_test_label_md01
            
            time_start = time.time()
            
            #n better components for model 01
            print('Extract features '+model_type[0]+"_"+model_dimension_reduction+"_"+str(number_reduce_components)+'f')
            dataset_train_md01, dataset_test_md01, time_reduction_md01 = feature_selection(dataset_train_md01, dataset_train_label_md01, dataset_test_md01, model_dimension_reduction, number_reduce_components)
            
            #n better components for model 02
            print('Extract features '+model_type[1]+"_"+model_dimension_reduction+"_"+str(number_reduce_components)+'f')
            dataset_train_md02, dataset_test_md02, time_reduction_md02 = feature_selection(dataset_train_md02, dataset_train_label_md02, dataset_test_md02, model_dimension_reduction, number_reduce_components)
            
            #concat n features each model
            dataset_train = np.hstack((dataset_train_md01,dataset_train_md02))
            dataset_test = np.hstack((dataset_test_md01,dataset_test_md02))
            
            
            
            time_reduction = time.time()-time_start
            
            print("feature selection: "+model_dimension_reduction+" time --> "+"{0:.4f}".format(time_reduction) + " \n")
            
            model_name = model_type[0]+"_"+str(number_reduce_components)+"f+"+model_type[1]+"_"+str(number_reduce_components)+"f"
            
            #classifier loop
            for model_classifier in model_classifier_list:
        
                if(model_classifier == 'PCC'):
                    dataset_train_PCC = np.vstack((dataset_train,dataset_test))
                    
                    #mask labels dataset test
                    mklabels = dataset_test_label.copy()
                    for x in range(len(mklabels)):
                        mklabels[x] = -1
                                               
                    #remount masked_labels                            
                    masked_labels = np.hstack((dataset_train_label,mklabels))                        
                
                    #run classifier
                    time_trainning, time_prediction, pred = classification_pcc(dataset_train_PCC,masked_labels,n_knn_neighbors)
                    #calc acc results
                    hidden_labels = dataset_test_label.copy()
                    hidden_pred = pred[masked_labels == -1]
                else:
                    time_trainning, time_prediction, pred = classification(dataset_train, dataset_train_label, dataset_test, model_classifier)
                    hidden_labels = dataset_test_label.copy()
                    hidden_pred = pred.copy()
                
                print(model_name + " >> " + model_dimension_reduction + " >> " + model_classifier + ': Kfold: '+str(index+1) + ' - Trainning --> '+"{0:.4f}".format(time_trainning) + ' - Prediction --> '+"{0:.4f}".format(time_prediction)+"\n")
                             
                if (model_classifier == 'PCC'):
                    print("PCC statistics")
                    #score's log
                    PCC_data_time_prediction.append(time_prediction)
                    PCC_data_time_trainning.append(time_trainning)
                    PCC_data_time_reduction.append(time_reduction)
                    PCC_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    PCC_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    PCC_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'J48'):
                    print("J48 statistics")
                    #score's log
                    J48_data_time_prediction.append(time_prediction)
                    J48_data_time_trainning.append(time_trainning)
                    J48_data_time_reduction.append(time_reduction)
                    J48_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    J48_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    J48_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                    
                elif (model_classifier == 'RBF'):
                    print("RBF statistics")
                    #score's log
                    RBF_data_time_prediction.append(time_prediction)
                    RBF_data_time_trainning.append(time_trainning)
                    RBF_data_time_reduction.append(time_reduction)
                    RBF_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    RBF_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    RBF_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'LinearSVM'):
                    print("LinearSVM statistics")
                    #score's log
                    LINEAR_data_time_prediction.append(time_prediction)
                    LINEAR_data_time_trainning.append(time_trainning)
                    LINEAR_data_time_reduction.append(time_reduction)
                    LINEAR_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    LINEAR_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    LINEAR_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'MLP'):
                    print("MLP statistics")
                    #score's log
                    MLP_data_time_prediction.append(time_prediction)
                    MLP_data_time_trainning.append(time_trainning)
                    MLP_data_time_reduction.append(time_reduction)
                    MLP_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    MLP_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    MLP_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'Logistic'):
                    print("Logistic statistics")
                    #score's log
                    LOG_data_time_prediction.append(time_prediction)
                    LOG_data_time_trainning.append(time_trainning)
                    LOG_data_time_reduction.append(time_reduction)
                    LOG_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    LOG_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    LOG_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'RandomForest'):
                    print("Random Forest statistics")
                    #score's log
                    FOREST_data_time_prediction.append(time_prediction)
                    FOREST_data_time_trainning.append(time_trainning)
                    FOREST_data_time_reduction.append(time_reduction)
                    FOREST_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    FOREST_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    FOREST_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'Adaboost'):
                    print("Adaboost statistics")
                    #score's log
                    ADA_data_time_prediction.append(time_prediction)
                    ADA_data_time_trainning.append(time_trainning)
                    ADA_data_time_reduction.append(time_reduction)
                    ADA_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    ADA_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    ADA_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                
                elif (model_classifier == 'Gaussian'):
                    print("Gaussian statistics")
                    #score's log
                    GAU_data_time_prediction.append(time_prediction)
                    GAU_data_time_trainning.append(time_trainning)
                    GAU_data_time_reduction.append(time_reduction)
                    GAU_acc_score.append(accuracy_score(hidden_labels,hidden_pred))
                    GAU_roc_score.append(roc_auc_score(hidden_labels,hidden_pred))
                    GAU_f1c_score.append(f1_score(hidden_labels,hidden_pred))
                    
                #csv detailed data
                print(" - Datailed statistics")
                with open(data_filename,"a+") as f_data:
                    f_data.write(model_name+",") #CNN
                    f_data.write(model_dimension_reduction+",") #Reduction_alg
                    f_data.write(model_classifier+",") #Classifier
                    f_data.write(str(index+1)+",") #Kfold index
                    f_data.write(str(number_features)+"," ) #CNN_features
                    f_data.write(scaled_feat_reduction+",") #Reduction_Scaled
                    f_data.write(str(np.shape(dataset_train)[1])+"," ) #Reduction_Components
                    if(model_classifier=="PCC"):
                        f_data.write(str(n_knn_neighbors)+",")  #k_neigh_PCC_classifier
                    else:
                        f_data.write(str(0)+",") 
                    f_data.write(str("{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)*100))+",") #Acc Score
                    f_data.write(str("{0:.4f}".format(f1_score(hidden_labels,hidden_pred)*100))+",") #F1 Score
                    f_data.write(str("{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)*100))+",") #ROC Score
                    f_data.write(str("{0:.4f}".format(time_feature_extration))+",") #Time Extraction Features
                    f_data.write(str("{0:.4f}".format(time_reduction))+",") #Time Reduction dimensionality
                    f_data.write(str("{0:.4f}".format(time_trainning))+",") #Time Classifier Trainning
                    f_data.write(str("{0:.4f}".format(time_prediction))+"\n") #Time Classifier Predict
                
                
                #PRINT ACCURACY SCORE
                print("\n Comp:" + str(number_reduce_components) + " -knn:" + str(n_knn_neighbors) + " Exec:" + str(index) + " - Acc Score:" + "{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)) + " f1 Score:" + "{0:.4f}".format(f1_score(hidden_labels,hidden_pred)) + " ROC Score:" + "{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)) + " Execution Time: " + "{0:.4f}".format(time_prediction+time_trainning) +'s')
                print("-------------------------------------------------------------------------------------------------------")

    

        #save logs
        for class_model in model_classifier_list:
            
            acc_score=[]
            f1c_score=[]
            roc_score=[]
            if (class_model == 'PCC'):
                acc_score = PCC_acc_score.copy()
                f1c_score = PCC_f1c_score.copy()
                roc_score = PCC_roc_score.copy()
             
            elif (class_model == 'J48'):
                acc_score = J48_acc_score.copy()
                f1c_score = J48_f1c_score.copy()
                roc_score = J48_roc_score.copy()
               
            elif (class_model == 'RBF'):
                acc_score = RBF_acc_score.copy()
                f1c_score = RBF_f1c_score.copy()
                roc_score = RBF_roc_score.copy()
            
            elif (class_model == 'LinearSVM'):
                acc_score = LINEAR_acc_score.copy()
                f1c_score = LINEAR_f1c_score.copy()
                roc_score = LINEAR_roc_score.copy()
            
            elif (class_model == 'MLP'):
                acc_score = MLP_acc_score.copy()
                f1c_score = MLP_f1c_score.copy()
                roc_score = MLP_roc_score.copy()
               
            elif (class_model == 'Logistic'):
                acc_score = LOG_acc_score.copy()
                f1c_score = LOG_f1c_score.copy()
                roc_score = LOG_roc_score.copy()
               
            elif (class_model == 'RandomForest'):
                acc_score = FOREST_acc_score.copy()
                f1c_score = FOREST_f1c_score.copy()
                roc_score = FOREST_roc_score.copy()
               
            elif (class_model == 'Adaboost'):
                acc_score = ADA_acc_score.copy()
                f1c_score = ADA_f1c_score.copy()
                roc_score = ADA_roc_score.copy()
              
            elif (class_model == 'Gaussian'):
                acc_score = GAU_acc_score.copy()
                f1c_score = GAU_f1c_score.copy()
                roc_score = GAU_roc_score.copy()
             
            #log acc
            with open(data_acc_filename,"a+") as f_acc_csv:
                f_acc_csv.write(model_name+",") #CNN
                f_acc_csv.write(model_dimension_reduction+",") #Reduction_alg
                f_acc_csv.write(scaled_feat_reduction+",")
                f_acc_csv.write(str(np.shape(dataset_train)[1])+"," ) #Reduction_Components
                f_acc_csv.write(class_model+",") #Classifier
                for acc in acc_score:
                    f_acc_csv.write("{0:.4f}".format(acc)+",")
                f_acc_csv.write("{0:.4f}".format(np.mean(acc_score))+",") 
                f_acc_csv.write("{0:.4f}".format(np.std(acc_score)))
                f_acc_csv.write("\n")
                        
            #log f1 score
            with open(data_f1_filename,"a+") as f_f1_csv:
                f_f1_csv.write(model_name+",") #CNN
                f_f1_csv.write(model_dimension_reduction+",") #Reduction_alg
                f_f1_csv.write(scaled_feat_reduction+",")
                f_f1_csv.write(str(np.shape(dataset_train)[1])+",") #Reduction_Components
                f_f1_csv.write(class_model+",") #Classifier
                for f1sc in f1c_score:
                    f_f1_csv.write("{0:.4f}".format(f1sc)+",")
                f_f1_csv.write("{0:.4f}".format(np.mean(f1c_score))+",") 
                f_f1_csv.write("{0:.4f}".format(np.std(f1c_score)))    
                f_f1_csv.write("\n")
                
            #log roc score
            with open(data_roc_filename,"a+") as f_roc_csv:
                f_roc_csv.write(model_name+",") #CNN
                f_roc_csv.write(model_dimension_reduction+",") #Reduction_alg
                f_roc_csv.write(scaled_feat_reduction+",")
                f_roc_csv.write(str(np.shape(dataset_train)[1])+"," ) #Reduction_Components
                f_roc_csv.write(class_model+",") #Classifier
                for roc_sc in roc_score:
                    f_roc_csv.write("{0:.4f}".format(roc_sc)+",")
                f_roc_csv.write("{0:.4f}".format(np.mean(roc_score))+",") 
                f_roc_csv.write("{0:.4f}".format(np.std(roc_score)))
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
#avg
#std




