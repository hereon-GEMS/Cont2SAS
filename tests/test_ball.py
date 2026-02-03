# Copyright (C) 2025  Helmholtz-Zentrum-Hereon

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

"""
test script for checking one time step model
or structure model

chosen model: ball

created by: Arnab Majumdar
Date      : 06.08.2025
"""

import shutil
import os
import subprocess
# import math
# import numpy as np
import pytest
# from matplotlib.testing.compare import compare_images as comp_img
import h5py

# non-test functions
def read_Iq_h5(Iq_data_h5, vol_norm):
    """
    Function for 
    1. Reading Iq data (Iq_data_h5)
    2. normalize using vol_norm
    """
    Iq_data=h5py.File(Iq_data_h5,'r')
    Iq=Iq_data['Iq'][:] # unit fm^2
    Iq=Iq/vol_norm*10**2 # unit (fm^2 / \AA^3) * 10^2 = cm^-1
    q=Iq_data['Q'][:]
    Iq_data.close()
    return q, Iq

# test functions

def test_ball_gen():
    """
    Test script for $C2S_HOME/models/ball/ball_gen.py:
    1. Check whether ball_gen runs
    """
    gen_result = subprocess.run(
        ["python", "models/ball/ball_gen.py"],
        capture_output=True,
        text=True,
        check=True
    )
    # Ensure it runs without crashing
    assert gen_result.returncode == 0, "ball model generates data"

def test_ball_plot():
    """
    Test script for $C2S_HOME/models/ball/ball_plot.py:
    1. Check whether ball_plot runs
    """
    plot_result = subprocess.run(
        ["python", "models/ball/ball_plot.py"],
        capture_output=True,
        text=True,
        check=True
    )
    # Ensure it runs without crashing
    assert plot_result.returncode == 0, "ball model plots data"

def test_check_ball_plot():
    """
    Test script for $C2S_HOME/models/ball/ball_plot.py:
    1. Check whether ball_plot plots IvsQ figure
    """
    plt_img='figure/ball/Iq_ball.pdf'
    if_exist=os.path.isfile(plt_img)
    # Ensure plotted file exist
    assert if_exist==1

# def test_compare_ball_plot():
#     """
#     Test script for $C2S_HOME/models/ball/ball_plot.py:
#     1. Compare plotted IvsQ figure with gold standard
#     """
#     gold_img='figure/gold/Iq_ball.pdf'
#     plt_img='figure/ball/Iq_ball.pdf'
#     # Check expected output
#     assert comp_img(plt_img, gold_img, tol=2) is None

def test_compare_ball_data():
    """
    Test script for $C2S_HOME/models/ball/ball_plot.py:
    1. Compare numerical IvsQ data with gold standard
    """
    # folder structure of gen data
    mesh_dir=os.path.join('data', '40p0_40p0_40p0_40_40_40_lagrangian_1')
    sim_dir=os.path.join(mesh_dir, 'simulation/ball_tend_0p0_dt_1p0_ensem_1')
    model_dir=os.path.join(sim_dir, 'rad_10_sld_2_qclean_sld_0')
    t_dir=os.path.join(model_dir, 't000')
    # data
    gold_data='tests/gold/ball_gold.h5'
    q_gold, Iq_gold=read_Iq_h5(gold_data, 1)
    num_data=os.path.join(t_dir, 'Iq_cat_extend_3Q_0p0_1p0_orien__10.h5')
    q_num, Iq_num=read_Iq_h5(num_data, 1)
    # compare num value and gold value
    assert q_num == pytest.approx(q_gold, abs=1e-6), "Q values match"
    assert Iq_num == pytest.approx(Iq_gold, abs=1e-6), "IQ values match"

def test_clean_up():
    """
    Test script for clean up:
    1. Removes created data and figure
    2. Checks whether removed
    """
    data_dir='data/40p0_40p0_40p0_40_40_40_lagrangian_1'
    fig_dir='figure/ball'
    # remove data files
    shutil.rmtree(data_dir)
    # remove figure
    shutil.rmtree(fig_dir)
    # check if it is removed
    assert os.path.isdir(data_dir)==0
    assert os.path.isdir(fig_dir)==0
