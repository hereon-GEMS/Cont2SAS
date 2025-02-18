#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework for saving data in hdf5 file
Created on Fri Jun 23 10:38:13 2023

@author: amajumda
"""
import sys
import os
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(lib_dir)


import h5py
import argparse, os
import numpy as np
import shutil
import datetime
import collections.abc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import math
from lib import analytical as ana
import xml.etree.ElementTree as ET

def fitter(Iq_num, q_num, model_xml, t_arr, t_idx, 
           fit_param_1, fit_param_2):
    t=t_arr[t_idx]
    t_end=np.max(t_arr)
    model=model_xml[-6:-4]
    if model=='gg':
        # read model xml
        tree=ET.parse(model_xml)
        root = tree.getroot()
        ball_sld=float(root.find('sld_in').text)
        box_sld=float(root.find('sld_out').text)
        sld_grain=ball_sld-box_sld
        r_0=float(root.find('rad_0').text)
        r_end=float(root.find('rad_end').text)
        print('fitting sphere')
        ball_rad_fit=fitter_gg(Iq_num, q_num, sld_grain)
        ball_rad_sim=r_0+((r_end-r_0)/t_end)*t
        print('ball radius simulation: {0}, fit: {1}'.format(ball_rad_sim, ball_rad_fit))
        fit_param_1[t_idx]=ball_rad_fit
        return fit_param_1, fit_param_2
    elif model=='fs':
        print('fitting fuzzy sphere')
        tree=ET.parse(model_xml)
        root = tree.getroot()
        ball_sld=float(root.find('sld_in').text)
        box_sld=float(root.find('sld_out').text)
        sld_grain=ball_sld-box_sld
        ball_rad=float(root.find('rad').text)
        sig_0=float(root.find('sig_0').text)
        sig_end=float(root.find('sig_end').text)
        sig_fit, ball_rad_fit=fitter_fs(Iq_num, q_num, sld_grain)
        sig_sim=sig_0+((sig_end-sig_0)/t_end)*t
        print('sigma simulation: {0}, fit: {1}'.format(sig_sim, sig_fit))
        print('radius simulation: {0}, fit: {1}'.format(ball_rad, ball_rad_fit))
        fit_param_1[t_idx]=sig_fit
        fit_param_2[t_idx]=ball_rad_fit
        return fit_param_1, fit_param_2
    else:
        print('not correct model')
           
def fitter_gg(Iq_num, q_num, sld_grain):
    def fit_func_gg(q_in, rad_opt):
        Iq, q_out =ana.ball(qmax=max(q_in),qmin=min(q_in),
                            Npts=len(q_in), scale=1 , bg=0,
                            sld=sld_grain, sld_sol=0,
                            rad=rad_opt)
        return Iq
    popt, pcov = curve_fit(fit_func_gg, q_num, Iq_num)
    # Iq_fit=fit_func(q_num, *popt_gg)
    rad_fit=popt[0]
    return rad_fit

def fitter_fs(Iq_num, q_num, sld_grain):
    def fit_func(q_in, sig_opt, rad_opt):
        Iq, q_out =ana.fuzzysph(qmax=max(q_in),qmin=min(q_in),
                            Npts=len(q_in), scale=1 , bg=0,
                            sld=sld_grain, sld_sol=0,
                            sig_fuzz=sig_opt, radius=rad_opt)
        return Iq
    popt, pcov = curve_fit(fit_func, q_num, Iq_num)
    # Iq_fit=fit_func(q_num, *popt)
    # plt.loglog(q_num, Iq_num)
    # plt.loglog(q_num, Iq_fit)
    # plt.show()
    # print('ball radius simulation: {0}, fit: {1}'.format(10, popt[1]))
    sig_fit=round(np.abs(popt[0]),2)
    ball_rad_fit=round(popt[1],2)
    return sig_fit, ball_rad_fit

def get_fit_param(model):
    if model=='gg':
        return 'radius'
    elif model=='fs':
        return ['fuzz_value', 'radius']