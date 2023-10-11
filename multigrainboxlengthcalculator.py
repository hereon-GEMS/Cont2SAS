#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:43:27 2023

@author: amajumda
"""
import numpy as np
import math
import matplotlib.pyplot as plt

r_med=35
sigma = 0.4#
#sigma=0.541*69
mu=np.log(r_med)
p=sigma*r_med

start=15#69-2*p
end=r_med+2*p

num_bin=10

#x=np.linspace(0.001,10*mu,num_bin)
x=np.linspace(start,end,num_bin)
#y=(1/sigma*np.sqrt(np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)
y=(1/x*sigma)*np.exp(-0.5*((np.log(x)-mu)/(sigma))**2)
y=y/np.sum(y)
plt.plot(x,100*y)
#plt.plot(np.log(x),y)

volume_faction=0.1

r=x
num_r=np.round(100*y)
#[num_r<1]
print('%%%%%% distribution summary %%%%%%')
for i in range(len(num_r)):
    print('radius: {0}, number: {1}'.format(r[i], num_r[i]))
    # print('number: {0}'.format(num_r[i]))

vol=((4/3)*np.pi*r**3)*num_r
vol_grain=np.sum(vol)
vol_total=vol_grain/volume_faction
boxlength=vol_total**(1/3)

print('simulation box length should be minimum {0}'.format(math.ceil(boxlength)) )