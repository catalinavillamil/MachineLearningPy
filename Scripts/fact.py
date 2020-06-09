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

def positivo(n):
    if n>=0:
        print("Es positivo")