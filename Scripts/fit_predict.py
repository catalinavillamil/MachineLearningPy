# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 07:29:31 2017

@author: Usuario
fit y prediccion
-- Implementado por clases de python
---- Tiene algoritmo de aprendizaje y predicción
---- Almacena la información aprendida
---- Entrena un modelo con el método fit()
---- Predice la nueva data con predict()
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris=datasets.load_iris()
knn=KNeighborsClassifier(n_neighbors=6)
iris['data']
iris['target']
knn.fit(iris['data'],iris['target'])
#
