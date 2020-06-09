# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 05:51:41 2017

@author: Usuario
"""
import pandas as pd
# combinación de datos
# Los datos no siempre puede venir en un archivo.
# Por ejemplo un data set de 5 millones de registros puede venir separado en 5 columnas
# Facil de almacenar y computar
# Muchos huecos en los datos cada día
# Importante para combinar datos que estén limpios
lista=['2015-06-01','2015-06-01','2015-06-02','2015-06-02']
lista1=['tmax','tmin','tmax','tmin']
lista2=[27.8,14.5,27.3,14.4]
data={'fecha':lista,'componente':lista1,'valor':lista2}
pf=pd.DataFrame(data)

lista=['2015-06-03','2015-06-03','2015-06-04','2015-06-04']
lista1=['tmax','tmin','tmax','tmin']
lista2=[27.8,14.5,27.3,14.4]
data={'fecha':lista,'componente':lista1,'valor':lista2}
pf1=pd.DataFrame(data)

datax=pd.concat([pf,pf1])
datax
datax=pd.concat([pf,pf1],ignore_index=True)
#datax
