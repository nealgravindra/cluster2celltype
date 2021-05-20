

import os
import glob
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns 
import matplotlib.pyplot as plt
import time 
import datetime
import pickle

from scipy.stats import zscore 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests

# settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=0.5
plt.rcParams['savefig.dpi']=600
sns.set_style("ticks")

pfp = '/home/ngr4/project/collabs/grants/czi_rp_2103/results/'
with open('/home/ngr4/project/collabs/grants/czi_rp_2103/data/processed/rpczi.pkl', 'rb') as f:
    temp = pickle.load(f)
    f.close()
adata = temp['adata']

# standard recipe
# sc.pp.combat(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', 'source'])

# define markers & cell types
## TODO (before final polishes): add canonical cell type markers for human lung from Table S1 https://www.biorxiv.org/content/10.1101/742320v2.full.pdf
## REF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5135277/ (most lung tissue markers come from here)
cell_markers = {'Basal': ['KRT5', 'DAPL1', 'TP63'],
                'Basal (proliferating)': ['ADH7', 'MKI67', 'TOP2A', 'CDK1'],
                'Hillock': ['KRT4', 'KRT13'],
                'Club': [ 'KRT15', 'CD74','CXCL6'],
                'Ciliated': ['FOXJ1', 'CCDC153', 'CCDC113', 'MLF1', 'LZTFL1','TUBB1','TP73','CCDC78'],
                'Tuft': ['POU2F3', 'AVIL', 'MAFF','MIAT','NOS2'],
                'Ionocyte': ['FOXI1', 'CFTR',], # 'ASCL3' not found
                'Goblet': ['MUC5AC', 'MUC5B', 'SPDEF'],
                'Epithelial':['ABCA3','LPCAT1','NAPSA','SFTPB','SFTPC','SLC34A2'],
                'Neuroendocrine':['ACADSB','ADA','AFAP1','CPE'],
                'Dendritic':['ITGAX','CCR7','CD1A','CD207'], # 'LY75' not found
#                 'Macrophage':['CD68','CD14','CCL18','CD163'],
                'Endothelial':['CD34','PECAM1','VWF'],
                'Fibroblast':['THY1','CD36','PDGFRA','PTPN13'],
                'Tcell':['CD3E','CD3D','CD3G','CD8A','CD8B','CD4'],
                'Granulocyte':['CCR5','SMAD1','ITGAM'],
#                 'Alveolar':['SLC34A2','ABCA3','CD44'],
                'AT1':['SLC34A2','ABCA3','CD44','AGER','PDPN','CLIC5'],
                'AT2':['SLC34A2','ABCA3','CD44','SFTPB','SFTPC','SFTPD','MUC1'],
                'Myofibroblast':['ACTA2'],
                'Monocyte':['CD36','CD14','CD68'],
                'NK':['NCR1'],
                'Progenitor':['TM4SF1','CEACAM6'],
#                 'Neutrophil':['S100A9','S100A8','S100A12','VCAN','FCN1',
#                               'CSTA','TSPO','CD14','MNDA','CTSD','PLBD1'], # from Tianyang (Iwasaki lab) ORIGINAL
                # updated 051820
                'Eosinophil':['RNASE2','LGALS1','RETN','AC020656.1', # 'RNASE3' not found
                              'H1FX','SLC44A1','AL355922.1','RFLNB','SERPINB10'], # from Tianyang (Iwasaki lab) ORIGINAL
#                 'Macrophage':['S100A9','S100A8','FCGR3A','CD14','CD68','FCGR1A','MARCO','MSR1','MRC1','C1QB','C1QA','FABP4','APOC1','APOE','PPARG'],
#                 'Monocyte':['S100A9','S100A8','FCGR3A','CD14','CD68','FCGR1A','RNASE2','RNASE3','FCN1','TNFRSF1B','S100A12','VCAN','CCR2','SDS'],
#                 'Monocyte':['CCR2', 'FCN1', 'RNASE2', 'RNASE3', 'S100A12', 'SDS', 'TNFRSF1B', 'VCAN'], # no overlap btw Macrophage/Monocyte/Neutrophil
                'Monocyte':['CCR2', 'FCN1', 'RNASE2', 'S100A12', 'SDS', 'TNFRSF1B', 'VCAN'],
                'Macrophage':['APOC1', 'APOE', 'C1QA', 'C1QB', 'FABP4', 'MARCO', 'MRC1', 'MSR1', 'PPARG'], # no overlap btw Macrophage/Monocyte/Neutrophil
                'Neutrophil':['CEACAM1', 'CEACAM8', 'CSF3R', 'CXCR1', 'CXCR2', 'FCGR3B'], # no overlap btw Macrophage/Monocyte/Neutrophil
#                 'Neutrophil':['S100A9','S100A8','FCGR3A','CEACAM8','CXCR1','CXCR2','CEACAM1','FCGR3B','CSF3R'],
#                 'Eosinophil':['RNASE2','RNASE3','IL5RA','CCR3','EPX','PRG2','PRG3','PTGDR2','SIGLEC8','GATA2'], # don't use RNASE2/3 since they overlap
#                 'Eosinophil':['IL5RA','CCR3','PRG2','PTGDR2','SIGLEC8','GATA2'], # don't use RNASE2/3 since they overlap
#                 'Eosinophil':['IL5RA','CCR3','PRG2','PTGDR2','SIGLEC8','GATA2', 'EPO','CD9','RNASE3','RETN','H1FX','RFLNB'], # added EPO and CD9 <>                
               }


# subset data to markers
genes = [g for k,v in cell_markers.items() for g in v]
x = pd.DataFrame(adata[:,genes].X, columns=genes)
x['cluster'] = adata.obs['leiden'].to_list()
add_pcs = True
if add_pcs:
    # add PCs?
    pcs = ['PC1','PC2']
    for i,pc in enumerate(pcs):       
        x[pc] = adata.obsm['X_pca'][:,i]
    genes = genes + pcs

# standard scale
x.loc[:,genes] = zscore(x.loc[:,genes])

results = pd.DataFrame()
fname = 'covid3balfs'

verbose = True
tic = time.time()
counter = 0
ORthreshold = 0.9
total_iter = len(cell_markers.keys())*len(x['cluster'].unique())
new_markers = {}
print('Lasso logistic regression')
for i,ctype in enumerate(cell_markers.keys()):
    for j,cluster in enumerate(x['cluster'].unique()):
        if verbose:
            if counter % 50 == 0 and counter != 0:
                p_through = counter / total_iter
                toc = time.time() - tic
                print('  through {:.1f}-% in {:.2f}-s\t~{:.2f}-s remain'.format(100*p_through,toc,(toc/counter)*(total_iter-counter)))
            
        # binarize & subset
        y = (x['cluster']==cluster).astype(int)
        if add_pcs:
            X = x.loc[:,cell_markers[ctype]+pcs]
        else:
            X = x.loc[:,cell_markers[ctype]]
        
        # run default params (could add CV)
        ## results, solver='saga', time for ~25k cells: >>1min
        ## results, solver='lbfgs', time for ~25k cells: 14s
        ## results, solver='liblinear', time for ~25k cells: 25s
        model = LogisticRegression(max_iter=10000, 
                                   penalty='l1',
                                   tol=1e-6,
                                   solver='liblinear') #n_jobs=-1 doesn't work for liblinear
        model.fit(X, y) 
        
        status = 'OK'
        if any(np.exp(model.coef_)[0][:-len(pcs)] < ORthreshold):
            markers = [marker for i,marker in enumerate(cell_markers[ctype]) if i not in np.where(np.exp(model.coef_)[0][:-len(pcs)]<0.9)[0]]
            if len(markers) != 0:
                new_markers[ctype] = markers
                if add_pcs:
                    X = x.loc[:,markers+pcs]
                else:
                    X = x.loc[:,markers]
                model = LogisticRegression(max_iter=10000, 
                                   penalty='l1',
                                   tol=1e-6,
                                   solver='liblinear') #n_jobs=-1 doesn't work for liblinear
                model.fit(X, y)
            else:
                status = 'No markers with ORs >= {}'.format(ORthreshold)
        else: 
            markers = cell_markers[ctype]
            
                
            
        
        p1 = model.predict_proba(X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y, p1)
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_pred = (p1>optimal_threshold).astype(int)
        precision,recall,_ = metrics.precision_recall_curve(y, p1)
        auprc = metrics.auc(recall, precision)
        auroc = metrics.roc_auc_score(y,p1)
        ap = metrics.average_precision_score(y,p1)
        bs = metrics.brier_score_loss(y,p1)
        acc = metrics.accuracy_score(y,optimal_pred)
        
        # store results
        dt = pd.DataFrame({'ctype2pred':ctype,
                           'cluster':cluster,
                           'auroc':auroc,
                           'status':status,
                           'markers':[markers],
                           'ORs':np.exp(model.coef_).tolist(),
                           'ave_prec':ap,
                           'acc':acc,
                           'sensitivity':tpr[optimal_idx],
                           'specificity':1-fpr[optimal_idx]},
                          index=[0])
        results = results.append(dt, ignore_index=True)
        counter += 1
print('Classifiers done. Saving and plotting...')  

top_per_ctype = pd.DataFrame()
top_n = 3
for ctype in results['ctype2pred'].unique():
    dt = results.loc[results['ctype2pred']==ctype,:]
    dt = dt.sort_values(by='auroc', ascending=False)
    top_per_ctype = top_per_ctype.append(dt.iloc[0:top_n,:], ignore_index=True)
    
top_per_cluster = pd.DataFrame()
top_n = 3
for cluster in results['cluster'].unique():
    dt = results.loc[results['cluster']==cluster,:]
    dt = dt.sort_values(by='auroc', ascending=False)
    top_per_cluster = top_per_cluster.append(dt.iloc[0:top_n,:], ignore_index=True)
    
if True:
    top_per_cluster.to_csv(os.path.join(pfp,'top_ctype_per_cluster_{}.csv'.format(fname)))
    
    
# plot init annotation
## taking top ctype per cluster
top1_per_cluster = pd.DataFrame()
for cluster in results['cluster'].unique():
    dt = results.loc[results['cluster']==cluster,:]
    dt = dt.sort_values(by='auroc', ascending=False)
    if True:
        # eliminate rows with poor status (no markers with OR>=threshold)
        dt = dt.loc[dt['status'] == 'OK',:]
    if dt.shape[0]==0:
        print('Cluster {} could not be annotated due to ORs of markers.'.format(cluster))
        continue
    top1_per_cluster = top1_per_cluster.append(dt.iloc[0:1,:], ignore_index=True)
ctype_annotation = {}
for cluster in top1_per_cluster['cluster']:
    ctype_annotation[cluster] = top1_per_cluster.loc[top1_per_cluster['cluster']==cluster,'ctype2pred'].values[0]
adata.obs['init_ctype'] = adata.obs['leiden'].astype(str)
adata.obs['init_ctype'] = adata.obs['init_ctype'].map(ctype_annotation)

## aesthetics
pal18=['#ee5264','#565656','#75a3b7','#ffe79e','#fac18a','#f1815f','#ac5861','#62354f','#2d284b','#f4b9b9','#c4bbaf',
               '#f9ebae','#aecef9','#aeb7f9','#f9aeae','#9c9583','#88bb92','#bde4a7','#d6e5e3']
cmap_ctype = {v:pal18[i] for i,v in enumerate(adata.obs['init_ctype'].unique())}

## plot
sc.pl.umap(adata, color=['leiden', 'init_ctype'])


if verbose:
    # print bad ones
    print('\nClusters hard to identify')
    print('-------------------------')
    
    underdetermined = top1_per_cluster.loc[top1_per_cluster['auroc'] <= 0.7, :]
    for i in range(underdetermined.shape[0]):
        print(underdetermined.iloc[i,:])
    
# save metadata
adata.obs.to_csv(os.path.join(pfp, 'metadata_{}'.format(datetime.datetime.now().strftime('%y%m%d'))))
