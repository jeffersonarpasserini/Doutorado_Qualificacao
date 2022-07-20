#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:16:40 2022

@author: jeffersonpasserini
"""

#gera arquivo de analise por método de extração de dados e red/sel. features.


from cmath import nan
import numpy as np
import pandas as pd

RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/"

df = pd.read_csv('results/data_detailed_alldata.csv', 
                names=['cnn','reduction','class','kfold','features','norm','components','pcc_knn','acc','f1','roc','extr_time','red_time','train_time','pred_time'], skiprows=1)

method = 'all'
method01 = ['Xception','VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152','ResNet50V2', 'ResNet101V2', 'ResNet152V2',
            'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'DenseNet121', 'DenseNet169','DenseNet201', 'NASNetMobile', 
            'MobileNetV2', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 
            'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

method02 = ['EfficientNetB1+EfficientNetB5','MobileNet+ResNet101','ResNet101+DenseNet169','ResNet101+DenseNet121','ResNet101+MobileNetV2',
            'EfficientNetB0+MobileNet','MobileNet+ResNet50','Xception+ResNet50','VGG16+VGG19']

method03 = ['EfficientNetB1+EfficientNetB5_n_feat','MobileNet+ResNet101_n_feat','ResNet101+DenseNet169_n_feat','ResNet101+DenseNet121_n_feat',
            'ResNet101+MobileNetV2_n_feat', 'EfficientNetB0+MobileNet_n_feat','MobileNet+ResNet50_n_feat',
            'Xception+ResNet50_n_feat','VGG16+VGG19_n_feat']

if (method=='method01'):
    model_type_list = method01
elif (method=='method02'):
    model_type_list = method02
elif (method=='method03'):
    model_type_list = method03
else:
    model_type_list = np.hstack((method01,method02,method03))

model_reduction_dim_list = ['PCA','UMAP','ReliefF']

model_classifier_list = ['PCC', 'J48', 'RBF', 'LinearSVM','MLP','Logistic','RandomForest','Adaboost','Gaussian']

components_number_list = [2,10,20,30,40,50,75,100,150,200,250,300,350,400,450,500,550,600,700,800,900,1000]

metric_type = 'acc'
distr_normal = 'no'

data_filename = RESULT_PATH+metric_type+"_components_resume_"+method+"_type2.csv"

with open(data_filename,"a+") as f_data:
    f_data.write('method,')
    f_data.write('cnn,') 
    f_data.write('reduction,') 
    f_data.write('classifier,')
    f_data.write('features,')
    f_data.write('Components,')
    f_data.write('acc,')
    f_data.write('f1,')
    f_data.write('roc'+"\n")
    
for extr_model in model_type_list:
    
    if (not extr_model.endswith('_n_feat')):
       
        for class_model in model_classifier_list:

            red_model = 'Full'
            print("Feat Extr: "+extr_model+" red/selection: "+red_model+" classifier: "+class_model)

            df_full = df.loc[(df['cnn']==extr_model) & (df['reduction']==red_model) & (df['class']==class_model)]
            if (distr_normal=='yes'):
                acc_full = df_full['acc'].mean()
                f1_full = df_full['f1'].mean()
                roc_full = df_full['roc'].mean()
            else:
                acc_full = df_full['acc'].median()
                f1_full = df_full['f1'].median()
                roc_full = df_full['roc'].median()

            #csv detailed data
            with open(data_filename,"a+") as f_data:
                f_data.write(extr_model+"_"+red_model+"_"+class_model+",") #method
                f_data.write(extr_model+",") #extracted method
                f_data.write(red_model+",") #Reduction_alg
                f_data.write(class_model+",") #Classifier
                f_data.write(str("{}".format(df_full['features'].mean()))+",") #number features extracted
                f_data.write('Full,') #number components
                f_data.write(str("{0:.4f}".format(acc_full))+",") #acc
                f_data.write(str("{0:.4f}".format(f1_full))+",") #f1
                f_data.write(str("{0:.4f}".format(roc_full))+"\n") #roc


    for red_model in model_reduction_dim_list:
        
        for class_model in model_classifier_list:
            
            print("Feat Extr: "+extr_model+" red/selection: "+red_model+" classifier: "+class_model)

            df_f = df.loc[(df['cnn']==extr_model) & (df['reduction']==red_model) & (df['class']==class_model)]
            
            for number_components in components_number_list:

                if (distr_normal=='yes'):
                    #componentes
                    acc = df_f.loc[(df_f['components']==number_components)]['acc'].mean()
                    f1 = df_f.loc[(df_f['components']==number_components)]['f1'].mean()
                    roc = df_f.loc[(df_f['components']==number_components)]['roc'].mean()
                else:
                    #componentes
                    acc = df_f.loc[(df_f['components']==number_components)]['acc'].median()
                    f1 = df_f.loc[(df_f['components']==number_components)]['f1'].median()
                    roc = df_f.loc[(df_f['components']==number_components)]['roc'].median()

                if (not np.isnan(acc)):                                       
                    #csv detailed data
                    with open(data_filename,"a+") as f_data:
                        f_data.write(extr_model+"_"+red_model+"_"+class_model+",") #method
                        f_data.write(extr_model+",") #extracted method
                        f_data.write(red_model+",") #Reduction_alg
                        f_data.write(class_model+",") #Classifier
                        f_data.write(str("{}".format(df_f['features'].mean()))+",") #number features extracted
                        f_data.write("F"+str("{}".format(number_components))+",") #number components
                        f_data.write(str("{0:.4f}".format(acc))+",") #acc
                        f_data.write(str("{0:.4f}".format(f1))+",") #f1
                        f_data.write(str("{0:.4f}".format(roc))+"\n") #roc
                
                