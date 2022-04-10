from numpy import linalg as LA
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA,KernelPCA,TruncatedSVD
import matplotlib.pyplot as plt

df = pd.read_csv('csv MNIST Dataset/train.csv', nrows = 20000)
df.head()
label = df['label']
df.drop('label', axis = 1, inplace = True)

data = df.to_numpy()

max_comp=783
start=1
error_record=[]

for i in range(start,max_comp):
    pca = PCA(n_components=i, random_state=42)
    pca2_results = pca.fit_transform(data)
    pca2_proj_back=pca.inverse_transform(pca2_results)
    total_loss=LA.norm((data-pca2_proj_back),None)
    error_record.append(total_loss)

plt.clf()
plt.figure(figsize=(15,15))
plt.title("reconstruct error of pca")
plt.plot(error_record,'r')
plt.xticks(range(len(error_record)), range(start,max_comp), rotation='vertical')
plt.xlim([-1, len(error_record)])
plt.show()
plt.savefig('mean reconstruction error',dpi=150)
