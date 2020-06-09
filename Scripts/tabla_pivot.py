# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 05:43:01 2017

@author: Usuario
"""
import pandas as pd
import numpy as np
##### Pivot_Table #####
#- Tiene una forma con la cual se puede trabajar con datos duplicados
# - Puede tomar el valor duplicado y calcular un estadístico ejemplo (Promedio)
lista=['2015-06-01','2015-06-01','2015-06-01','2015-06-01']
lista1=['tmax','tmin','tmax','tmin']
lista2=[27.8,14.5,27.3,14.4]
data={'fecha':lista,'componente':lista1,'valor':lista2}
pf=pd.DataFrame(data)
pf_tidy=pf.pivot_table(values='valor',index='fecha',columns='componente',aggfunc=np.mean)
#pf_tidy=pf.pivot_table(values='valor',index='componente',columns='fecha',aggfunc=np.mean)
