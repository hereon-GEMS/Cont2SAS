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
def read_Iq_h5(Iq_data_h5):
    """
    Function for 
    1. Reading Iq data (Iq_data_h5)
    """
    Iq_data=h5py.File(Iq_data_h5,'r')
    Iq=Iq_data['Iq'][:]
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
    Test script for $C2S_HOME/models/phase_field/phase_field_gen.py:
    1. Compare numerical IvsQ data with gold standard
    """
    # folder structure of gen data
    ## meshing dir
    mesh_dir=os.path.join('data', '250p0_250p0_250p0_100_100_100_lagrangian_1')
    mesh_exist=os.path.isdir(mesh_dir)
    assert mesh_exist==1, "meshing dir exist"
    ## sim time = 0s
    ## C2S creates model for each simulated time step
    ## each simulated time step then have one C2S time step
    ## this algo was in place because C2S only allows for linspace time arr
    ### model dir 1
    sim_dir=os.path.join(mesh_dir, 'simulation/phase_field_tend_0p0_dt_1p0_ensem_1')
    model_dir_1=os.path.join(sim_dir, 'name_spinodal_fe_cr_time_0_qclean_sld_0p5410058695982137')
    model_exist_1=os.path.isdir(model_dir_1)
    assert model_exist_1==1, "model dir exist, sim time 0"
    ### time dir 1
    t_dir_1=os.path.join(model_dir_1, 't000')
    t_exist_1=os.path.isdir(t_dir_1)
    assert t_exist_1==1, "time dir exist, sim time 0"
    ### data
    #### num data (read + check exist)
    num_data_1=os.path.join(t_dir_1, 'Iq_cat_extend_501Q_0p025132741228718346_1p0_orien__200.h5')
    data_exist_1=os.path.isfile(num_data_1)
    assert data_exist_1==1, "hdf5 file exist, sim time 0"
    q_num_1, Iq_num_1=read_Iq_h5(num_data_1)
    #### gold data
    gold_data_1='tests/gold/phase_field_gold_1.h5'
    q_gold_1, Iq_gold_1=read_Iq_h5(gold_data_1)
    ## sim time = 86400s
    ### model dir 2
    ### sim dir is same
    model_dir_2=os.path.join(sim_dir, 'name_spinodal_fe_cr_time_86400_qclean_sld_0p5408560149631366')
    model_exist_2=os.path.isdir(model_dir_2)
    assert model_exist_2==1, "model dir exist, sim time 86400"
    ### time dir 2
    t_dir_2=os.path.join(model_dir_2, 't000')
    t_exist_2=os.path.isdir(t_dir_2)
    assert t_exist_2==1, "time dir exist, sim time 86400"
    ### data
    #### num data (read + check exist)
    num_data_2=os.path.join(t_dir_2, 'Iq_cat_extend_501Q_0p025132741228718346_1p0_orien__200.h5')
    data_exist_2=os.path.isfile(num_data_2)
    assert data_exist_2==1, "hdf5 file exist, sim time 86400"
    q_num_2, Iq_num_2=read_Iq_h5(num_data_2)
    #### gold data
    gold_data_2='tests/gold/phase_field_gold_2.h5'
    q_gold_2, Iq_gold_2=read_Iq_h5(gold_data_2)
    # compare num value and gold value
    ## sim time = 0s
    assert q_num_1 == pytest.approx(q_gold_1, abs=1e-6), "Q values match, t=0s"
    assert Iq_num_1 == pytest.approx(Iq_gold_1, abs=1e-6), "IQ values match, t=0s"
    ## sim time = 86400s
    assert q_num_2 == pytest.approx(q_gold_2, abs=1e-6), "Q values match, t=86400s"
    assert Iq_num_2 == pytest.approx(Iq_gold_2, abs=1e-6), "IQ values match, t=86400s"

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
