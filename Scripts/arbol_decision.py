# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 23:43:47 2017

@author: Usuario
"""
from sklearn import tree 
import pandas as pd
import numpy as np

edad=[10,20,30,40,29]
sexo=['F','M','F','M','F']
sobre=[0.4,0.3,0.1,0.2,0.1]
data={'edad':edad,'sexo':sexo,'sobre':sobre}
data=pd.DataFrame(data)
target=data['sobre'].values
features=[['edad','sexo']]
mi_arbol=tree.DecisionTreeClassifier()
mi_arbol=mi_arbol.fit(target,features)
print(mi_arbol.feature_importances_)
#print(my_tree_one.score(, ___))
