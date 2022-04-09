import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

df = pd.read_csv('train.csv', nrows = 20000)

#print("the shape of data is :", df.shape)

df.head()

label = df['label']
df.drop('label', axis = 1, inplace = True)
ind = np.random.randint(0, 20000)
#plt.figure(figsize = (20, 5))
grid_data = np.array(df.iloc[ind]).reshape(28,28)

def Reconstruct(data_matrix,k):
    plt.imshow(grid_data)
    plt.show()
    img_mat_scaled= (grid_data-grid_data.mean())/grid_data.std()
    U, s, V = np.linalg.svd(img_mat_scaled) 
    
    var_explained = np.round(s**2/np.sum(s**2), decimals=3)
    
    sns.barplot(x=list(range(1,21)),
    y=var_explained[0:20], color="dodgerblue")
    plt.xlabel('Singular Vector', fontsize=16)
    plt.ylabel('Variance Explained', fontsize=16)
    plt.tight_layout()

    #plt.savefig('svd_scree_plot.png',dpi=150)
    num_components = k
    reconst_img_k = np.matrix(U[:, :num_components]) * np.diag(s[:num_components]) *np.matrix(V[:num_components, :])
    plt.imshow(reconst_img_k)
    plt.show()
    #plt.savefig('reconstructed_image_with_1_SVs.png',dpi=150)
    
Reconstruct(grid_data,6)

#plt.imshow(grid_data, interpolation = None, cmap = 'gray')
#plt.show()
#print(label[ind]))




