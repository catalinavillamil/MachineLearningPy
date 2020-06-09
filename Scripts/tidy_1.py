# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 04:47:22 2017

@author: Usuario
"""
import pandas as pd
# Introducción a Tidy Data - Fue propuesta por Hadye Wickham 
#- Es una manera de formalizar los datos
#- En caso de no tenerlo permite formatearlo
#- Es una forma de organizar nuestros data set
#- Sus principios son:
#- Su columnas representan la separación de variables
#- Las filas representan las observaciones individuales
#- La unidades de análsis son las tablas
lista=['carlos','jose','federico']
lista1=['',12,24]
lista2=[42,31,27]
data1={'nombre':lista,'tratamiento_1':lista1,'tratamiento_2':lista2}
pf=pd.DataFrame(data1)
# - Preguntas tipicas:
# - Mejor para reportar vs Mejor Organizar 
#- Hace más fácil la reparación de errores
# - Los problemas más comunes arreglando datos son:
#-Las columnas tienen datos en vez de variables.
# - Una forma adecuada de reparar el usando pd.melt()
n=pd.melt(pf, id_vars=['nombre'], value_vars=['tratamiento_1','tratamiento_2'])
n
pd.melt(pf, id_vars=['nombre'], value_vars=['tratamiento_1','tratamiento_2'],var_name=['tratamiento'],value_name=['Resultado'])
n.to_csv('tidy1.csv')