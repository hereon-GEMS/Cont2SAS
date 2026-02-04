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
test script for checking multi time step model
or phenomena model

chosen model: sld grow

created by: Arnab Majumdar
Date      : 06.08.2025
"""

import shutil
import os
import subprocess
import pytest
# from matplotlib.testing.compare import compare_images as comp_img
import h5py

# non-test functions
def read_sig_eff_h5(sig_eff_data_h5):
    """
    Function for 
    1. Reading sig_eff data (sig_eff_data_h5)
    2. normalize using vol_norm
    """
    sig_eff_data=h5py.File(sig_eff_data_h5,'r')
    sig_eff=sig_eff_data['sig_eff'][:]
    t=sig_eff_data['t'][:]
    sig_eff_data.close()
    return t, sig_eff

# test functions
def test_sld_grow_gen():
    """
    Test script for $C2S_HOME/models/sld_grow/sld_grow_gen.py:
    1. Check whether sld_grow_gen runs
    """
    gen_result = subprocess.run(
        ["python", "models/sld_grow/sld_grow_gen.py"],
        capture_output=True,
        text=True,
        check=True
    )
    # Ensure it runs without crashing
    assert gen_result.returncode == 0, "sld_grow model generates data"

def test_sld_grow_plot():
    """
    Test script for $C2S_HOME/models/sld_grow/sld_grow_plot.py:
    1. Check whether sld_grow_plot runs
    """
    plot_result = subprocess.run(
        ["python", "models/sld_grow/sld_grow_plot.py"],
        capture_output=True,
        text=True,
        check=True
    )
    # Ensure it runs without crashing
    assert plot_result.returncode == 0, "sld_grow model plots data"

def test_check_sld_grow_plot():
    """
    Test script for $C2S_HOME/models/sld_grow/sld_grow_plot.py:
    1. Check whether sld_grow_plot plots sig_effvst figure
    """
    plt_img='figure/sld_grow/sig_eff_fit_sld_grow.pdf'
    if_exist=os.path.isfile(plt_img)
    # Ensure plotted file exist
    assert if_exist==1

# def test_compare_sld_grow_plot():
#     """
#     Test script for $C2S_HOME/models/sld_grow/sld_grow_plot.py:
#     1. Compare plotted IvsQ figure with gold standard
#     """
#     gold_img='figure/gold/sig_eff_fit_sld_grow.pdf'
#     plt_img='figure/sld_grow/sig_eff_fit_sld_grow.pdf'
#     # Check expected output
#     assert comp_img(plt_img, gold_img, tol=2) is None

def test_compare_sld_grow_data():
    """
    Test script for $C2S_HOME/models/sld_grow/sld_grow_gen.py:
    1. Compare numerical sig_eff vs t data with gold standard
    """
    # folder structure of gen data
    ## meshing dir
    mesh_dir=os.path.join('data', '40p0_40p0_40p0_40_40_40_lagrangian_1')
    mesh_exist=os.path.isdir(mesh_dir)
    assert mesh_exist==1, "meshing dir exist"
    ## model dir
    sim_dir=os.path.join(mesh_dir, 'simulation/sld_grow_tend_10p0_dt_1p0_ensem_1')
    model_dir=os.path.join(sim_dir, 'rad_10_sld_in_0_2_sld_in_end_5_sld_out_1_qclean_sld_1')
    model_exist=os.path.isdir(model_dir)
    assert model_exist==1, "model dir exist"
    # data
    ## num data (read + check exist)
    num_data=os.path.join(model_dir, 'sig_eff_cat_extend_101Q_0p0029_0p05_orien__10.h5')
    data_exist=os.path.isfile(num_data)
    assert data_exist==1, "hdf5 file exist"
    t_num, sig_eff_num=read_sig_eff_h5(num_data)
    ## gold data
    gold_data='tests/gold/sld_grow_gold.h5'
    t_gold, sig_eff_gold=read_sig_eff_h5(gold_data)
    # compare num value and gold value
    assert t_num == pytest.approx(t_gold, abs=1e-6), "t values match"
    assert sig_eff_num == pytest.approx(sig_eff_gold, abs=1e-6), "sig eff values match"

def test_clean_up():
    """
    Test script for clean up:
    1. Removes created data and figure
    2. Checks whether removed
    """
    data_dir='data/40p0_40p0_40p0_40_40_40_lagrangian_1'
    fig_dir='figure/sld_grow'
    # remove data files
    shutil.rmtree(data_dir)
    # remove figure
    shutil.rmtree(fig_dir)
    # check if it is removed
    assert os.path.isdir(data_dir)==0
    assert os.path.isdir(fig_dir)==0
