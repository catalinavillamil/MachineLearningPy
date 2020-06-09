# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 07:15:23 2017

@author: Usuario
## K-nearest Neighbors
El objetivo principal es predecir la etiqueta de un dato por un punto:
-- Mirar en el k vencidades la etiqueta de los puntos.
-- Tomar la mayor votación
-- Ej: Petalos
"""
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris=datasets.load_iris()
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
df.head()
plt.plot(df['petal width (cm)'],df['petal length (cm)'],'or')
plt.show()