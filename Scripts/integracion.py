# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 21:56:44 2017

@author: Usuario
"""

""" La integración es una alternativa  con la
cual se puede reducir la inconsistencia y redundacia de la información

- La heterogeneidad semántica y la estructura de datos posee
grandes cambios en la integración.

- Redundancia y análisis de correlación
-- Test de Correlación para datos nominales (Chi2)
-- Test de Correlación para datos (númericos)
-- Covarianza de datos númericos
-- Reducción de datos 
--- Reducción de dimensión (PCA,Gráfica,)
--- Reducción númerica 
--- Compresión de datos
"""
import pandas as pd
import numpy as np

# Similar al Join de tablas en SQL
# Combina dataset dispares basados en columnas comunes

np.random.seed(12345)
dep=['Valle del Cauca','Cundinamarca','Antioquia','Atlantico']
sigla=['VLL','CUN','ANT','ATL']
ciudad=['Cali','Bogotá','Medellín','Barranquilla']
data={'Departamento':dep,'Poblacion':np.random.uniform(1000000,9000000,size=len(dep))}
data1={'Dep':dep,'Siglas':sigla}
data2={'Departamento':dep,'Ciudad':ciudad}
data=pd.DataFrame(data)
data1=pd.DataFrame(data1)
data2=pd.DataFrame(data2)
#d1=data.merge(data1)
w=pd.merge(left=data,right=data1,on=None,left_on='Departamento',right_on='Dep')
w=pd.merge(left=data,right=data2,on='Departamento')