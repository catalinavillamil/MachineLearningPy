# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 06:56:01 2017

@author: Usuario

--- El aprendizaje supervisado trabaja con datasets y sklearn
posee algunos de prueba: como el iris dataset

### Características: Tamaño del petalo, ancho del petalo, sépalo (ancho, largo)
### Target: Especie: Versicolo, Virginica, Setosa
"""
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris=datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data),type(iris.target)
iris.data.shape
iris.target_names
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
pd.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')
