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
from matplotlib.testing.compare import compare_images as comp_img

def test_ball_gen():
    """
    Test script for $C2S_HOME/models/ball/ball_gen.py:
    1. Check whether ball_gen runs
    """
    gen_result = subprocess.run(
        ["python", "models/ball/ball_gen.py"], 
        capture_output=True, 
        text=True
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
        text=True
    )
    # Ensure it runs without crashing
    assert plot_result.returncode == 0, "ball model plots data"

def test_check_ball_plot():
    """
    Test script for $C2S_HOME/models/ball/ball_plot.py:
    1. Check whether ball_plot plots IvsQ figure
    """
    gold_img='figure/gold/Iq_ball.pdf'
    plt_img='figure/ball/Iq_ball.pdf'
    if_exist=os.path.isfile(plt_img)
    # Ensure plotted file exist
    assert if_exist==1

def test_compare_ball_plot():
    """
    Test script for $C2S_HOME/models/ball/ball_plot.py:
    1. Compare plotted IvsQ figure with gold standard
    """
    gold_img='figure/gold/Iq_ball.pdf'
    plt_img='figure/ball/Iq_ball.pdf'
    # Check expected output
    assert comp_img(plt_img, gold_img, tol=2)==None

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

