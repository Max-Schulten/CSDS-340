#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:42:42 2025

@author: maximilianschulten
"""

import matplotlib.pyplot as plt

class1 = [(1,1), (1,2), (2,1)]

class2 = [(0,0), (1,0), (0,1)]


plt.figure(figsize=(6,6))
plt.scatter([x for x,y in class1], [y for x,y in class1], c='blue', label='Class +1')
plt.scatter([x for x,y in class2], [y for x,y in class2], c='red', label='Class -1')

plt.xlabel("x1")
plt.ylabel("x2")

plt.show()