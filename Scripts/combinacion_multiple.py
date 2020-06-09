# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 06:01:17 2017

@author: Usuario
"""
#Combinación de varios
# Manejo de grandes datos, busca patrones de almacenamiento
import glob
import pandas as pd
x=glob.glob('*.csv')
x
d=list()
for i in x:
    l=pd.read_csv(i)
    d.append(l)
f=pd.concat(d)
#f
