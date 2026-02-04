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
test script for checking functional model
or simulated model

chosen model: phase_field

created by: Arnab Majumdar
Date      : 06.08.2025
"""

import shutil
import os
import subprocess
from matplotlib.testing.compare import compare_images as comp_img
import pytest
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

def test_phase_field_gen():
    """
    Test script for $C2S_HOME/models/phase_field/phase_field_gen.py:
    1. Check whether phase_field_gen runs
    """
    gen_result = subprocess.run(
        ["python", "models/phase_field/phase_field_gen.py"],
          capture_output=True,
            text=True,
              check=True
    )
    # Ensure it runs without crashing
    assert gen_result.returncode == 0, "phase_field model generates data"

def test_phase_field_plot():
    """
    Test script for $C2S_HOME/models/phase_field/phase_field_plot.py:
    1. Check whether phase_field_plot runs
    """
    plot_result = subprocess.run(
        ["python", "models/phase_field/phase_field_plot.py"],
          capture_output=True,
            text=True,
              check=True
    )
    # Ensure it runs without crashing
    assert plot_result.returncode == 0, "phase_field model plots data"

def test_check_phase_field_plot():
    """
    Test script for $C2S_HOME/models/phase_field/phase_field_plot.py:
    1. Check whether phase_field_plot plots IvsQ figure
    """
    plt_img='figure/phase_field/Iq_phase_field.pdf'
    if_exist=os.path.isfile(plt_img)
    # Ensure plotted file exist
    assert if_exist==1

def test_compare_phase_field_plot():
    """
    Test script for $C2S_HOME/models/phase_field/phase_field_plot.py:
    1. Compare plotted IvsQ figure with gold standard
    """
    gold_img='figure/gold/Iq_phase_field.pdf'
    plt_img='figure/phase_field/Iq_phase_field.pdf'
    # Check expected output
    assert comp_img(plt_img, gold_img, tol=2) is None

def test_compare_phase_field_data():
    """
    Test script for $C2S_HOME/models/phase_field/phase_field_plot.py:
    1. Compare numerical IvsQ data with gold standard
    """
    # folder structure of gen data
    mesh_dir=os.path.join('data', '250p0_250p0_250p0_100_100_100_lagrangian_1')
    sim_dir=os.path.join(mesh_dir, 'simulation/phase_field_tend_0p0_dt_1p0_ensem_1')
    model_dir1=os.path.join(sim_dir, 'name_spinodal_fe_cr_time_0_qclean_sld_0p5410058695982137')
    t_dir1=os.path.join(model_dir1, 't000')
    model_dir2=os.path.join(sim_dir, 'name_spinodal_fe_cr_time_86400_qclean_sld_0p5408560149631366')
    t_dir2=os.path.join(model_dir2, 't000')
    # data 1
    gold_data_1='tests/gold/phase_field_gold_1.h5'
    q_gold_1, Iq_gold_1=read_Iq_h5(gold_data_1, 1)
    num_data_1=os.path.join(t_dir1, 'Iq_cat_extend_501Q_0p025132741228718346_1p0_orien__200.h5')
    q_num_1, Iq_num_1=read_Iq_h5(num_data_1, 1)
    # compare num value and gold value
    assert q_num_1 == pytest.approx(q_gold_1, abs=1e-6), "Q values match"
    assert Iq_num_1 == pytest.approx(Iq_gold_1, abs=1e-6), "IQ values match"
    # data 2
    gold_data_2='tests/gold/phase_field_gold_2.h5'
    q_gold_2, Iq_gold_2=read_Iq_h5(gold_data_2, 1)
    num_data_2=os.path.join(t_dir2, 'Iq_cat_extend_501Q_0p025132741228718346_1p0_orien__200.h5')
    q_num_2, Iq_num_2=read_Iq_h5(num_data_2, 1)
    # compare num value and gold value
    assert q_num_2 == pytest.approx(q_gold_2, abs=1e-6), "Q values match"
    assert Iq_num_2 == pytest.approx(Iq_gold_2, abs=1e-6), "IQ values match"

def test_clean_up():
    """
    Test script for clean up:
    1. Removes created data and figure
    2. Checks whether removed
    """
    data_dir='data/250p0_250p0_250p0_100_100_100_lagrangian_1'
    fig_dir='figure/phase_field'
    # remove data files
    shutil.rmtree(data_dir)
    # remove figure
    shutil.rmtree(fig_dir)
    # check if it is removed
    assert os.path.isdir(data_dir)==0
    assert os.path.isdir(fig_dir)==0
