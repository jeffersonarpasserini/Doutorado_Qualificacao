#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:16:40 2022

@author: jeffersonpasserini
"""
import pandas as pd

RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/"

df = pd.read_csv('results/data_daitaled.csv', 
                names=['cnn','reduction','class','kfold','features','norm','components','pcc_knn','acc','f1','roc','extr_time','red_time','train_time','pred_time'], skiprows=1)

model_type_list = ['MobileNet+ResNet101_n_feat','ResNet101+DenseNet169_n_feat','ResNet101+DenseNet121_n_feat','ResNet101+MobileNetV2_n_feat',
                   'EfficientNetB0+MobileNet_n_feat','MobileNet+ResNet50_n_feat','Xception+ResNet50_n_feat','VGG16+VGG19_n_feat',
                   'MobileNet+ResNet101','ResNet101+DenseNet169','ResNet101+DenseNet121','ResNet101+MobileNetV2',
                   'EfficientNetB0+MobileNet','MobileNet+ResNet50','Xception+ResNet50','VGG16+VGG19', 'Xception', 
                   'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                   'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169',
                   'DenseNet201', 'NASNetMobile', 'MobileNetV2',
                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
                   'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                   'EfficientNetB6', 'EfficientNetB7']


model_reduction_dim_list = ['PCA','UMAP','ReliefF']

model_classifier_list = ['PCC', 'J48', 'RBF', 'LinearSVM','MLP','Logistic','RandomForest','Adaboost','Gaussian']


metric_type = 'acc'
distr_normal = 'yes'

data_filename = RESULT_PATH+"method_components_"+metric_type+"_resume.csv"

with open(data_filename,"a+") as f_data:
    f_data.write('method,')
    f_data.write('cnn,') 
    f_data.write('reduction,') 
    f_data.write('classifier,')
    f_data.write('features,')
    f_data.write('F001,')
    f_data.write('F010,')
    f_data.write('F020,')
    f_data.write('F030,')
    f_data.write('F040,')
    f_data.write('F050,')
    f_data.write('F075,')
    f_data.write('F100,')
    f_data.write('F150,')
    f_data.write('F200,')
    f_data.write('F250,')
    f_data.write('F300,')
    f_data.write('Full,')
    f_data.write('Mean,')
    f_data.write('Sd,')
    f_data.write('MedianQ2,')
    f_data.write('Q1_25%,')
    f_data.write('Q3_75%,')
    f_data.write('Max,')
    f_data.write('Min \n')
    

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
            Full = df_full['acc'].mean()
            
            #resultado_final
            #juntar dataframe full + dataframe do method redim atual.
            #para considerar na mean and median
            
            df_concat = pd.concat((df_f,df_full))
            
            df_mean = df_concat.describe()
            method_mean = df_mean.loc['mean'][metric_type]
            method_mean_std  = df_mean.loc['std'][metric_type]
            method_median = df_mean.loc['50%'][metric_type]
            method_median_25 = df_mean.loc['25%'][metric_type]
            method_median_75 = df_mean.loc['75%'][metric_type]
            method_max = df_mean.loc['max'][metric_type]
            method_min = df_mean.loc['min'][metric_type]
            
            #csv detailed data
            with open(data_filename,"a+") as f_data:
                f_data.write(extr_model+"_"+red_model+"_"+class_model+",") #method
                f_data.write(extr_model+",") #extracted method
                f_data.write(red_model+",") #Reduction_alg
                f_data.write(class_model+",") #Classifier
                f_data.write(str("{}".format(df_f['features'].mean()))+",") #number features extracted
                f_data.write(str("{0:.4f}".format(F001))+",") #1 component
                f_data.write(str("{0:.4f}".format(F010))+",") #10 components
                f_data.write(str("{0:.4f}".format(F020))+",") #20 components
                f_data.write(str("{0:.4f}".format(F030))+",") #30 components
                f_data.write(str("{0:.4f}".format(F040))+",") #40 components
                f_data.write(str("{0:.4f}".format(F050))+",") #50 components
                f_data.write(str("{0:.4f}".format(F075))+",") #75 components
                f_data.write(str("{0:.4f}".format(F100))+",") #100 components
                f_data.write(str("{0:.4f}".format(F150))+",") #150 components
                f_data.write(str("{0:.4f}".format(F200))+",") #200 components
                f_data.write(str("{0:.4f}".format(F250))+",") #250 components
                f_data.write(str("{0:.4f}".format(F300))+",") #300 components
                f_data.write(str("{0:.4f}".format(Full))+",") #Full components
                f_data.write(str("{0:.4f}".format(method_mean))+",") #mean
                f_data.write(str("{0:.4f}".format(method_mean_std))+",") #std
                f_data.write(str("{0:.4f}".format(method_median))+",") #median Q2 - 50%
                f_data.write(str("{0:.4f}".format(method_median_25))+",") #Q1 - 1st quarter 25%
                f_data.write(str("{0:.4f}".format(method_median_75))+",") #Q3 - 1st quarter 25%
                f_data.write(str("{0:.4f}".format(method_max))+",") #max
                f_data.write(str("{0:.4f}".format(method_min))+"\n") #min
                