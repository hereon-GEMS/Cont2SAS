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
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3))
    
    if cols == 1:
        axes = [axes]
    
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    
    fig.tight_layout()
    fig.savefig(output_image, dpi=dpi)
    plt.close(fig)

# Example usage
# mother dir
fig_dir='./joss_submission/figures/'
# create figure with three pdfs
pdf_files = ['Iq_sld_grow.pdf', 'sig_eff_fit_sld_grow.pdf']
for pdf_idx, pdf_file in enumerate(pdf_files):
    pdf_files[pdf_idx]=fig_dir + pdf_file
out_fig=fig_dir + 'sig_eff.png'
create_pdf_figure(pdf_files, out_fig)