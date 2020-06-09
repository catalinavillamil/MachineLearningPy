# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 06:19:45 2017

@author: Usuario
"""

## Uso de expresiones regulares 
# Muchos datos pueden ser manipulados por 
# 17   \d*
# $17  \$ \d*
# $17.00 \$\d*.\d*
#$17.89 \$\d*\.\d{2}
# 17.895 
import re
patron=re.compile('\$\d*.\d{2}')
r=patron.match('$17')
bool(r)

#df.apply(col,axis=1,pattern=pat)