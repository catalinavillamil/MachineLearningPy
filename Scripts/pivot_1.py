# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 05:20:46 2017

@author: Usuario
"""
import pandas as pd
# - Opuesto al Milting
#- En Melt se rotan entre filas y columnas.
# - Pivoteo: Rotar un unico valor en las columnas separadas
# - Se analiza de forma amigable, reportando la información de manera amigable
#- Variables multiples almacenadas en la misma columna

lista=['2015-05-01','2015-07-01','2015-08-01','2015-09-01']
lista1=['tmax','tmin','tmax','tmin']
lista2=[27.8,14.5,27.3,14.4]
data={'fecha':lista,'componente':lista1,'valor':lista2}
pf=pd.DataFrame(data)

#pf_tidy=pf.pivot(index='fecha',columns='componente',values='valor')
#pf_tidy.to_csv('tidy_2.csv')
#- No se pueden usar entrada duplicadas