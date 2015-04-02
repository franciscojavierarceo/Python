# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:35:56 2015

@author: franciscojavierarceo
"""

import pandas as pd
df = pd.DataFrame()
df['x'] = range(0,10)
df['y'] = range(10,20)
df['z'] = df['x'] + df['y']
print df

x = df['x'] 
y = df['y']
z = x + y
print x,y,z
