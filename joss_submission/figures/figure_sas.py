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

from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def create_pdf_figure(pdf_paths, output_image='figure.png', dpi=150):
    images = []
    for path in pdf_paths:
        pages = convert_from_path(path, dpi=dpi)
        images.append(pages[0])  # use first page only

    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    
    if cols == 1:
        axes = [axes]
    
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_image, dpi=dpi)
    plt.close(fig)

def create_mesh_figure(length_a, length_b, nx, ny,
                        output_pdf='mesh.pdf', dpi=150):
    # fig initialization
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    # plot point so that add patch works
    ax.plot([0],[0], 'k')
    # plot mesh
    cell_x=length_a/nx
    cell_y=length_b/ny
    for idx1 in range(nx):
        for idx2 in range(ny):
            rect_center_x=idx1*cell_x
            rect_center_y=idx2*cell_y
            rect=Rectangle((rect_center_x, rect_center_y), cell_x, cell_y, 
                                           edgecolor='k', facecolor='none', linewidth=0.5)
            ax.add_patch(rect)
    # save figure
    fig.savefig(output_pdf, format='pdf', dpi=dpi)
    plt.close(fig)

# Example usage
# mother dir
fig_dir='./joss_submission/figures/'
# mesh params
length_a= 40.  
length_b=40.
# number of cells in each direction (int values)
nx= 40
ny= 40
# create mesh figure
create_mesh_figure(length_a, length_b, nx, ny, output_pdf= fig_dir + 'mesh.pdf')
# create figure with three pdfs
pdf_files = ['mesh.pdf', 'SLD_ball.pdf', 'Iq_ball.pdf']
for pdf_idx, pdf_file in enumerate(pdf_files):
    pdf_files[pdf_idx]=fig_dir + pdf_file
out_fig=fig_dir + 'workflow.png'
create_pdf_figure(pdf_files, out_fig)