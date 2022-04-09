import pandas as pd
import numpy as np
from PIL import Image
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('train.csv', nrows = 2000)

#df_y = pd.DataFrame(df[df.columns[0]])
#df_x = pd.DataFrame(df[df.columns[1:785]])

df_x = df.iloc[:,1:]
#print(df_x)
df_y = df.iloc[:,0]

svd = TruncatedSVD(n_components=10)
x = svd.fit(df_x).transform(df_x)

x_train,x_test,y_train,y_test = train_test_split(x,df_y,test_size=0.2,random_state=4)

rf = RandomForestClassifier(n_estimators = 50)
rf.fit(x_train,y_train)

pred = rf.predict(x_test)
s = y_test.values

print(len(svd.components_))
#print(svd.components_)
#for i in range(10):
#    array = svd.components_[i]
#    global_contrast_normalization(array, 1, 10, 0.000000001)
#    array = np.reshape(array, (28, 28))
#    data = Image.fromarray(array)
#    data.show()
#    print(array)
    
