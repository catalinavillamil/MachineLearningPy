# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:36:18 2017

@author: Usuario
"""

import numpy as np
import scipy.stats.norm
a=0
b=1



lambda_etapa=[]
etapas=0
x=list()
t=list()
sumas=list()
trabajadores=10
for i in  range(trabajadores):
 alfa_i=0
 p=0
 xi=0 
 etapas=1
 
 while p <= alfa_i:
  t.append(i)
  alfa_i=np.random.rand()
  p=np.random.rand()  
  lamb=(np.random.rand()*(b-a))+a
  w=(-lamb*(np.log(np.random.rand()))) 
  x.append(w)
  xi=xi+w
  etapas=etapas + 1 
 sumas.append(xi/(etapas-1))
print 'fin'  