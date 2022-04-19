# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:27:23 2021

@author: jeffe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


choice = 'umap' # umap, pca, geral
plot_graph ='line' #heatmap dispersion line histogram

if (choice=='pca'):
    #----------- Importação de Dados - PCA
    pca_avg_std = pd.read_csv('results_pca_euclidean/resultado.csv', names=['comp','knn','acc','acc_std','f_score','f_score_std','roc','roc_std','time','time_std'], skiprows=1)
    pca_result = pd.read_csv('results_pca_euclidean/resultado_geral.csv', names=['knn', 'comp', 'acc','f_score','roc','time'], skiprows=1)
    
    
    
    # PCA Heatmap
    if (plot_graph == 'heatmap'):
        pca_acc_array = pd.DataFrame({'comp': pca_avg_std['comp'], 
                                'knn': pca_avg_std['knn'], 
                                'acc': pca_avg_std['acc']})
                
        
        pca_pivot = pca_acc_array.pivot('comp','knn','acc')
        sns.set(font_scale=0.8)
        ax = sns.heatmap(pca_pivot, cmap='hot', linewidth=0.5, annot=True, fmt='.4f')
        plt.xlabel('PCC k-vizinhos', fontsize=10)
        plt.ylabel('PCA Componentes', fontsize=10)
        plt.gcf().set_size_inches(20,10)
        plt.show()
    
    
    
    if(plot_graph == 'histogram'):
        pca_result_acc = pd.DataFrame({'comp': pca_result['comp'], 
                                'knn': pca_result['knn'], 
                                'acc': pca_result['acc']
                                })
        
        
        sns.set(font_scale=1)
        ax = sns.lineplot(x='knn', y='acc', kind='box', data=pca_result_acc[pca_result_acc['comp']==7])
        plt.xlabel('PCC k-vizinhos', fontsize=20)
        plt.ylabel('Acurácia', fontsize=20)
        plt.gcf().set_size_inches(20,10)
        plt.show()
        
        
        sns.set(font_scale=1)
        ax = sns.lineplot(x='comp', y='acc', kind='box', data=pca_result_acc[pca_result_acc['knn']==25])
        plt.xlabel('PCA p-componentes', fontsize=20)
        plt.ylabel('Acurácia', fontsize=20)
        plt.gcf().set_size_inches(20,10)
        plt.show()
        
        
        
        
    if(plot_graph == 'line'):
        pca_result_acc = pd.DataFrame({'comp': pca_avg_std['comp'], 
                                'knn': pca_avg_std['knn'], 
                                'acc': pca_avg_std['acc']})

        sns.set(font_scale=1)
        ax = sns.lineplot(x='knn', y='acc', data=pca_result_acc[pca_result_acc['comp']==7])
        plt.xlabel('PCC k-vizinhos (p-componentes = 7)', fontsize=10)
        plt.ylabel('Acurácia', fontsize=10)
        plt.gcf().set_size_inches(20,10)
        plt.show()
      
        sns.set(font_scale=1)
        ax = sns.lineplot(x='comp', y='acc', data=pca_result_acc[pca_result_acc['knn']==25])
        plt.xlabel('PCA p-componentes (knn=25)', fontsize=10)
        plt.ylabel('Acurácia', fontsize=10)
        plt.gcf().set_size_inches(20,10)
        plt.show()
        
        
    
    


if (choice=='umap'):
    #----------- Importação de Dados - UMAP
    umap_avg_std = pd.read_csv('results_umap_euclidean/resultado.csv', names=['comp','knn','acc','acc_std','f_score','f_score_std','roc','roc_std','time','time_std'], skiprows=1)
    umap_result = pd.read_csv('results_umap_euclidean/resultado_geral.csv')
    
    
    # UMAP Heatmap
    if (plot_graph == 'heatmap'):
        umap_acc_array = pd.DataFrame({'comp': umap_avg_std['comp'], 
                                'knn': umap_avg_std['knn'], 
                                'acc': umap_avg_std['acc']})
        
        umap_pivot = umap_acc_array.pivot('comp','knn','acc')
        sns.set(font_scale=0.8)
        ax = sns.heatmap(umap_pivot, cmap='hot', linewidth=0.5, annot=True, fmt='.4f')
        plt.xlabel('PCC k-vizinhos', fontsize=10)
        plt.ylabel('UMAP Componentes', fontsize=10)
        plt.gcf().set_size_inches(20,10)
        plt.show()
        
        
    if(plot_graph == 'line'):
        
        knn_p = np.vstack((umap_avg_std[umap_avg_std['comp']==21],umap_avg_std[umap_avg_std['comp']==24]))
        
        pca_result_acc_novo = pd.DataFrame({'comp': knn_p[:,0],
                                            'knn': knn_p[:,1], 
                                            'acc': knn_p[:,2]})
        
        pca_result_acc_novo["comp"] = pca_result_acc_novo["comp"].replace({21: 'p=21', 24: 'p=24'}) 
        
        sns.set(font_scale=1)
        ax = sns.lineplot(x='knn', y='acc', hue='comp', data=pca_result_acc_novo)
        plt.xlabel('PCC k-vizinhos', fontsize=10)
        plt.ylabel('Acurácia', fontsize=10)
        plt.gcf().set_size_inches(20,10)
        plt.show()
        
        
        knn_p = np.vstack((umap_avg_std[umap_avg_std['knn']==4], umap_avg_std[umap_avg_std['knn']==5],
                           umap_avg_std[umap_avg_std['knn']==6],umap_avg_std[umap_avg_std['knn']==15]))
        
        pca_result_acc_novo = pd.DataFrame({'comp': knn_p[:,0],
                                            'knn': knn_p[:,1], 
                                            'acc': knn_p[:,2]})
        
        pca_result_acc_novo["knn"] = pca_result_acc_novo["knn"].replace({4: 'k=4', 5: 'k=5', 6: 'k=6', 15: 'k=15'}) 
        
        sns.set(font_scale=1)
        ax = sns.lineplot(x='comp', y='acc', hue='knn', data=pca_result_acc_novo)
        plt.xlabel('PCC p-componentes', fontsize=10)
        plt.ylabel('Acurácia', fontsize=10)
        plt.gcf().set_size_inches(20,10)
        plt.show()
        
    
    
    
if(choice == 'geral'):
    
    #grafico dispersão tempo x acc -    
    umap_geral = pd.read_csv('results_umap_euclidean/resultado_geral.csv', names=['knn', 'comp', 'acc','f_score','roc','time'], skiprows=1) 
    pca_geral = pd.read_csv('results_pca_euclidean/resultado_geral.csv', names=['knn', 'comp', 'acc','f_score','roc','time'], skiprows=1)

    tipos = np.hstack((np.full((31250),'pca'),np.full((31250),'umap')))
    comp = np.hstack((pca_geral['comp'].to_numpy(),umap_geral['comp'].to_numpy()))
    knn = np.hstack((pca_geral['knn'].to_numpy(),umap_geral['knn'].to_numpy()))
    acc = np.hstack((pca_geral['acc'].to_numpy(),umap_geral['acc'].to_numpy()))
    time = np.hstack((pca_geral['time'].to_numpy(),umap_geral['time'].to_numpy()))
    
    pca_result_acc = pd.DataFrame({'type': tipos.tolist(),'comp': comp.tolist(), 'knn': knn.tolist(),'acc': acc.tolist(), 'time': time.tolist() })
    
    
    sns.set_style("white")
    plt.figure(figsize=(30, 30))
    g = sns.relplot(x='acc', y='time', hue='type', data=pca_result_acc[pca_result_acc['type']=='pca'], palette=['b'])
    plt.show()
    
    sns.set_style("white")
    plt.figure(figsize=(30, 30))
    g = sns.relplot(x='acc', y='time', hue='type', data=pca_result_acc[pca_result_acc['type']=='umap'], palette=['r'])
    plt.show()
    
    sns.set_style("white")
    plt.figure(figsize=(30, 30))
    g = sns.relplot(x='acc', y='time', hue='type', data=pca_result_acc, palette=['b','r'])
    plt.show()
