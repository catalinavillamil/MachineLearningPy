# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:23:50 2017

@author: Usuario
"""

import numpy as np




def factorial(n):
  fac=1
  for i in range(1,n+1):
      fac=fac*i
  return fac


a=1 # jsdsmdmsdm
b=5
c=12

x1=0
X2=0
w=np.sqrt((b**2)-(4*a*c))
if w>=0:
    x1=(-b+w)/(2*a)
    x2=(-b-w)/(2*a)
elif w==-1:
 print("Es -1")    
elif w<0:
  print("No es real")


if x1 in np.arange(0,10,1):
    print("ok")
lista=['a','b','c']
for i in range(len(lista)):
   if i==1:
       break
   print(lista[i])
    
for i in lista:
    print(i)
    
while a==1:
 while True:
     