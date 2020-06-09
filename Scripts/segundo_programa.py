# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:47:00 2017

@author: Usuario
"""

import numpy as np


numdias=30
ventas=np.zeros((numdias))
ventas[0]=32
np.random.seed(32)

def calculo():
 global ganancia
 p=0
 for i in range(1,numdias):
    al=np.random.rand()
    if al <= 0.05:
        ventas[i]=30
    elif al>0.05 and al<=0.20:
        ventas[i]=31
    elif al>0.20 and al<=0.42:
        ventas[i]=32
    elif al>0.42 and al<=0.80:
        ventas[i]=33
    elif al>0.80 and al<=0.94:
        ventas[i]=34
    elif al>0.94 and al<=1:
        ventas[i]=35
 dif=list()  
 ganancia=list()      
 ganancia.append(ventas[0]*600 - (ventas[0]*400))
 for i in range(1,numdias):
    a=ventas[i]-ventas[i-1]
    if a < 0:
        p=np.abs(a*200)
    elif a > 0:
        p=a*100
    elif p==0:
        p=0
    ganancia.append(ventas[i]*600 - ((ventas[i]*400)+p))
    dif.append(ventas[i]-ventas[i-1])

ganancia=list()
for i in range(120):
 calculo()
 print np.mean(ganancia)
    