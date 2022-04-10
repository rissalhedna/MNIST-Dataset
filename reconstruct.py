import matplotlib.pyplot as plt
from fastai.vision.all import *
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import torch

#First part, specific to MNIST dataset:
df = pd.read_csv('csv MNIST Dataset/train.csv', nrows = 20000)
df.head()
label = df['label']
df.drop('label', axis = 1, inplace = True)
ind = np.random.randint(0, 20000)
#plt.figure(figsize = (20, 5))
grid_data = np.array(df.iloc[ind]).reshape(28,28)

#Function for plotting variance explained by each principal component
def PlotVariance(var_explained):
    sns.barplot(x=list(range(1,21)),
    y=var_explained[0:20], color="dodgerblue")
    plt.xlabel('Singular Vector', fontsize=16)
    plt.ylabel('Variance Explained', fontsize=16)
    plt.show()
    plt.savefig('svd_scree_plot.png',dpi=150)

    
#The function needed:
def Reconstruct(data_matrix,k):
    img_mat_scaled= (grid_data-grid_data.mean())/grid_data.std()
    U, s, V = np.linalg.svd(img_mat_scaled) 
    var_explained = np.round(s**2/np.sum(s**2), decimals=3)
    plt.tight_layout()
    num_components = k
    reconst_img_k = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) *np.matrix(V[:num_components, :])
    plt.imshow(reconst_img_k)
    plt.show()
    M = torch.mean(Tensor(reconst_img_k).float())
    print("d-dimensional mean:",M)
    return var_explained
    
Reconstruct(grid_data,1)
Reconstruct(grid_data,2)
Reconstruct(grid_data,3)
Reconstruct(grid_data,4)
Reconstruct(grid_data,5)
Reconstruct(grid_data,6)
Reconstruct(grid_data,7)
Reconstruct(grid_data,8)
Reconstruct(grid_data,9)
var_explained = Reconstruct(grid_data,10)

#Calling PlotVariance to get the variance explained
PlotVariance(var_explained)


