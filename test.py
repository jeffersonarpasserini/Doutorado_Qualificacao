#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:22:26 2022

@author: jeffersonpasserini
"""

RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/"
data_filename = RESULT_PATH+"method_components_acc_resume.csv"

df = pd.read_csv('/home/jeffersonpasserini/dados/ProjetosPos/Doutorado_Qualificacao/results/data_daitaled.csv', names=['cnn','reduction','class','kfold','features','norm','components','pcc_knn','acc','f1','roc','extr_time','red_time','train_time','pred_time','column'], skiprows=1)

df_full = df.loc[(df['cnn']=='Xception+ResNet50') & (df['reduction']=='Full') & (df['class']=='PCC')]

df_f = df.loc[(df['cnn']=='Xception+ResNet50') & (df['reduction']=='ReliefF') & (df['class']=='PCC')]

F001 = df_f.loc[(df_f['components']==1)]['acc'].mean()
F010 = df_f.loc[(df_f['components']==10)]['acc'].mean()
F020 = df_f.loc[(df_f['components']==20)]['acc'].mean()
F030 = df_f.loc[(df_f['components']==30)]['acc'].mean()
F040 = df_f.loc[(df_f['components']==40)]['acc'].mean()
F050 = df_f.loc[(df_f['components']==50)]['acc'].mean()
F075 = df_f.loc[(df_f['components']==75)]['acc'].mean()
F100 = df_f.loc[(df_f['components']==100)]['acc'].mean()
F150 = df_f.loc[(df_f['components']==150)]['acc'].mean()
F200 = df_f.loc[(df_f['components']==200)]['acc'].mean()
F250 = df_f.loc[(df_f['components']==250)]['acc'].mean()
F300 = df_f.loc[(df_f['components']==300)]['acc'].mean()
Full = df_full['acc'].mean()

df_concat = pd.concat((df_f,df_full))
df_mean = df_concat.describe()

method_mean = df_mean.loc['mean']['acc']
method_mean_std  = df_mean.loc['std']['acc']
method_median = df_mean.loc['50%']['acc']
method_median_25 = df_mean.loc['25%']['acc']
method_median_75 = df_mean.loc['75%']['acc']
method_max = df_mean.loc['max']['acc']
method_min = df_mean.loc['min']['acc']
 
extr_model = 'Xception+ResNet50'
red_model = 'ReliefF'
class_model = 'PCC'
 
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