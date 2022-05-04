#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:16:40 2022

@author: jeffersonpasserini
"""
import numpy as np
import pandas as pd

RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/"

df = pd.read_csv('results/data_daitaled.csv', 
                names=['cnn','reduction','class','kfold','features','norm','components','pcc_knn','acc','f1','roc','extr_time','red_time','train_time','pred_time','column'], skiprows=1)

model_type_list = ['Xception+ResNet50','VGG16+VGG19', 'Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 
        'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2', "InceptionV3",
        'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169',
        'DenseNet201', 'NASNetMobile', 'MobileNetV2',
        'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
        'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
        'EfficientNetB6', 'EfficientNetB7']

model_reduction_dim_list = ['PCA','UMAP','ReliefF']

model_classifier_list = ['PCC', 'J48', 'RBF', 'LinearSVM','MLP','Logistic','RandomForest','Adaboost','Gaussian']


metric_type = 'acc'
distr_normal = 'yes'

data_filename = RESULT_PATH+"method_components_"+metric_type+"_resume.csv"

for extr_model in model_type_list:
    for red_model in model_reduction_dim_list:
        
        
        for class_model in model_classifier_list:
            
            print("Feat Extr: "+extr_model+" red/selection: "+red_model+" classifier: "+class_model)

            df_full = df.loc[(df['cnn']==extr_model) & (df['reduction']=='Full') & (df['class']==class_model)]
            df_f = df.loc[(df['cnn']==extr_model) & (df['reduction']==red_model) & (df['class']==class_model)]
            
            
            #componentes
            F001 = df_f.loc[(df_f['components']==1)][metric_type].mean()
            F010 = df_f.loc[(df_f['components']==10)][metric_type].mean()
            F020 = df_f.loc[(df_f['components']==20)][metric_type].mean()
            F030 = df_f.loc[(df_f['components']==30)][metric_type].mean()
            F040 = df_f.loc[(df_f['components']==40)][metric_type].mean()
            F050 = df_f.loc[(df_f['components']==50)][metric_type].mean()
            F075 = df_f.loc[(df_f['components']==75)][metric_type].mean()
            F100 = df_f.loc[(df_f['components']==100)][metric_type].mean()
            F150 = df_f.loc[(df_f['components']==150)][metric_type].mean()
            F200 = df_f.loc[(df_f['components']==200)][metric_type].mean()
            F250 = df_f.loc[(df_f['components']==250)][metric_type].mean()
            F300 = df_f.loc[(df_f['components']==300)][metric_type].mean()
            Full_mean = df_full.mean()
            
            #resultado_final
            #juntar dataframe full + dataframe do method redim atual.
            #para considerar na mean and median
            
            #frames = (df_f2,df_f)
            #df_concat = pd.concat(frames)
            
            df_mean = df_f.describe()
            method_mean = desc_teste.loc['mean'][metric_type]
            method_std  = desc_teste.loc['std'][metric_type]
            
            
            
            #csv detailed data
            with open(data_filename,"a+") as f_data:
                f_data.write(extr_model+"_"+red_model+"_"+class_model+",") #method
                f_data.write(extr_model+",") #extracted method
                f_data.write(red_model+",") #Reduction_alg
                f_data.write(class_model+",") #Classifier
                f_data.write(str("{}".format(df_f['features'].mean()))+",")
                f_data.write(str("{0:.4f}".format(F001))+",")
                f_data.write(str("{0:.4f}".format(F010))+",")
                f_data.write(str("{0:.4f}".format(F020))+",")
                f_data.write(str("{0:.4f}".format(F030))+",")
                f_data.write(str("{0:.4f}".format(F040))+",")
                f_data.write(str("{0:.4f}".format(F050))+",")
                f_data.write(str("{0:.4f}".format(F075))+",")
                f_data.write(str("{0:.4f}".format(F100))+",")
                f_data.write(str("{0:.4f}".format(F150))+",")
                f_data.write(str("{0:.4f}".format(F200))+",")
                f_data.write(str("{0:.4f}".format(F250))+",")
                f_data.write(str("{0:.4f}".format(F300))+",")
                f_data.write(str("{0:.4f}".format(df_mean))+"\n")
                