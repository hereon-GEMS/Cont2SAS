#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework for saving data in hdf5 file
Created on Fri Jun 23 10:38:13 2023

@author: amajumda
"""

import numpy as np

# Auxilliary functions
def J1(x):
    if x==0:
        return 0
    else:
        return (np.sin(x)-x*np.cos(x))/x**2

def arrange_order(a,b,c):
    zusammen=[a,b,c]
    zusammen.sort()
    return zusammen[0], zusammen[1], zusammen[2]

def gauss_legendre_double_integrate(func, domain1, domain2, deg):
    x, w = np.polynomial.legendre.leggauss(deg)
    xgrid, ygrid=np.meshgrid(x,x)
    x=xgrid.reshape((1,np.size(xgrid)))
    y=ygrid.reshape((1,np.size(ygrid)))
    wx, wy=np.meshgrid(w,w)
    w1=wx.reshape((1,np.size(wx)))
    w2=wy.reshape((1,np.size(wy)))
    s1 = (domain1[1] - domain1[0])/2
    a1 = (domain1[1] + domain1[0])/2
    s2 = (domain2[1] - domain2[0])/2
    a2 = (domain2[1] + domain2[0])/2
    return np.sum(s1*s2*w1*w2*func(s1*x + a1,s2*y + a2))

# Analytical functions for different shapes

def ball (qmax,qmin,Npts,scale,bg,sld,sld_sol,rad):
    vol=(4/3)*np.pi*rad**3
    # SLD unit 10^-5 \AA^-2
    del_rho=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=vol*del_rho
        else:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=3*vol*del_rho*J1(q*rad)/(q*rad)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = ((scale)*np.abs(FormFactor)**2+bg)
    return Iq_arr, q_arr

def fuzzysph(qmax,qmin,Npts,scale,bg,sld,sld_sol,sig_fuzz,radius):
    vol=(4/3)*np.pi*radius**3
    # SLD unit 10^-5 \AA^-2
    del_rho=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts)
    FormFactor=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=vol*del_rho
        else:
            # Form factor unit 10^-5 \AA
            FormFactor[i]=(3*vol*del_rho*J1(q*radius)/(q*radius))*np.exp((-(sig_fuzz*q)**2)/2)
    # Intensity unit 10^-10 \AA^2
    Iq_arr = scale*np.abs(FormFactor)**2+bg
    return Iq_arr, q_arr

def box (qmax,qmin,Npts,scale,bg,sld,sld_sol,length_a,length_b,length_c):
    length_a, length_b, length_c= arrange_order(length_a,length_b,length_c)
    vol_box=length_a*length_b*length_c
    # SLD unit 10^-5 \AA^-2
    del_rho_box=sld-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts) 
    Aq_arr=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        func=lambda alpha, psi:\
            (del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
            (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
            (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha))))**2*\
            np.sin(alpha)
            
        psi_lim=np.pi
        alpha_lim=np.pi/2 
        # Amplitude unit (\AA^3 * 10^-5 \AA^-2)^2 = 10^-10 \AA^2
        Aq_arr[i]=(1/psi_lim)*gauss_legendre_double_integrate(func,[0, alpha_lim],[0, psi_lim],76)
    Iq_arr = scale*Aq_arr + bg # Intensity unit 10^-10 \AA^2
    return Iq_arr, q_arr

def ball_in_box(qmax,qmin,Npts,scale,scale2,bg,sld_box, sld_ball,sld_sol,length_a,length_b,length_c,radius):
    length_a, length_b, length_c=arrange_order(length_a,length_b,length_c)
    vol_box=length_a*length_b*length_c
    vol_ball=(4/3)*np.pi*radius**3
    # SLD unit 10^-5 \AA^-2
    del_rho_box=sld_box-sld_sol
    del_rho_ball=sld_ball-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts) 
    Aq_arr=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            func=lambda alpha, psi:\
                (del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha)))-\
                scale2*1+\
                scale2*1)**2*\
                np.sin(alpha)
        else:
            func=lambda alpha, psi:\
                (del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha)))-\
                scale2*3*vol_ball*del_rho_box*J1(q*radius)/(q*radius)+\
                scale2*3*vol_ball*del_rho_ball*J1(q*radius)/(q*radius))**2*\
                np.sin(alpha)
        
        psi_lim=np.pi
        alpha_lim=np.pi/2 
        # Amplitude unit (\AA^3 * 10^-5 \AA^-2)^2 = 10^-10 \AA^2
        Aq_arr[i]=(1/psi_lim)*gauss_legendre_double_integrate(func,[0, alpha_lim],[0, psi_lim],76)
    Iq_arr = scale*Aq_arr + bg # Intensity unit 10^-10 \AA^2
    return Iq_arr, q_arr

def ball_in_box_ecc(qmax,qmin,Npts,scale,scale2,bg,sld_box, sld_ball,sld_sol,
                    length_a,length_b,length_c,radius, origin_shift):
    length_a, length_b, length_c=arrange_order(length_a,length_b,length_c)
    vol_box=length_a*length_b*length_c
    vol_ball=(4/3)*np.pi*radius**3
    # SLD unit 10^-5 \AA^-2
    del_rho_box=sld_box-sld_sol
    del_rho_ball=sld_ball-sld_sol
    q_arr=np.linspace(qmin,qmax,Npts) 
    Aq_arr=np.zeros(len(q_arr))
    for i in range(len(q_arr)):
        q=q_arr[i]
        if q==0:
            func=lambda alpha, psi:\
                np.abs(del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha))))**2*\
                    np.sin(alpha)
        else:
            func=lambda alpha, psi:\
                np.abs(del_rho_box*vol_box*(np.sinc((1/np.pi)*q*length_a/2*np.sin(alpha)*np.sin(psi)))*\
                (np.sinc((1/np.pi)*q*length_b/2*np.sin(alpha)*np.cos(psi)))*\
                (np.sinc((1/np.pi)*q*length_c/2*np.cos(alpha)))-\
                scale2*3*vol_ball*(del_rho_box-del_rho_ball)*\
                    J1(q*radius)/(q*radius)*\
                    np.vectorize(complex)(np.cos(q * np.sin(alpha) * np.sin(psi) * origin_shift[0]\
                                    + q * np.sin(alpha) * np.cos(psi) * origin_shift[1]\
                                        + q *  np.cos(psi) * origin_shift[2]),
                            np.sin(q * np.sin(alpha) * np.sin(psi) * origin_shift[0]\
                                    + q * np.sin(alpha) * np.cos(psi) * origin_shift[1]\
                                        + q *  np.cos(psi) * origin_shift[2])))**2*\
                    np.sin(alpha)
        
        psi_lim=np.pi
        alpha_lim=np.pi/2
        # Amplitude unit (\AA^3 * 10^-5 \AA^-2)^2 = 10^-10 \AA^2 
        Aq_arr[i]=(1/psi_lim)*gauss_legendre_double_integrate(func,[0, alpha_lim],[0, psi_lim],76)
    Iq_arr = scale*Aq_arr + bg # Intensity unit 10^-10 \AA^2    
    return Iq_arr, q_arr