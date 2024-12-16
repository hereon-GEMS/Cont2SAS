
import numpy as np
from scipy import special
from scipy.optimize import minimize
#from numba import njit, prange
from scipy.special import erf
import xml.etree.ElementTree as ET
import os


def pseudo_b(node_sld,connec,cell_vol):
    num_node=len(node_sld)
    num_cell=len(connec)
    pseudo_b=np.zeros(num_cell)
    for i in range(num_cell):
        cell_i_node_idx = connec[i,:]
        cell_i_node_sld = node_sld[cell_i_node_idx]
        cell_i_cell_sld = np.average(cell_i_node_sld)
        pseudo_b[i] = cell_vol*cell_i_cell_sld
    return pseudo_b

def pseudo_b_cat(pseudo_b,num_cat,method='extend'):
    # sorting of b_values
    b_sort=np.sort(pseudo_b)
    b_arg_sort=np.argsort(b_sort)
    b_min=b_sort[0]
    b_max=b_sort[-1]
    b_range=b_max-b_min
    # no categorization
    if num_cat==0:
        return pseudo_b
    else:
        if method=='simple':
            """
            range of b values
            [o......o......o      ...      o] 
              div1    div2    div3...  divN
            |o......|o......|o    ...      o|
            """
            cat_min=b_min
            cat_max=b_max
            cat_width=(cat_max-cat_min)/num_cat
            cat_val_arr=np.linspace(cat_min+cat_width/2, cat_max-cat_width/2, num_cat)
            cat_range_arr=np.linspace(cat_min,cat_max,num_cat+1)    
            b_sort_cat=np.zeros(len(b_sort))
            sort_cat=np.zeros(len(b_sort))
            for i in range(len(cat_val_arr)):
                # lower and upper bound of current category in sorted b array
                cat_low_bound=cat_range_arr[i]
                cat_low_bound_arg=len(b_sort[b_sort<cat_low_bound])
                cat_upper_bound=cat_range_arr[i+1]
                cat_upper_bound_arg=len(b_sort[b_sort<=cat_upper_bound])
                # cut sorted b array between lower and upper bound of current category
                # reset categorical b array values to b value of category
                # reset category array values to current category
                b_sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=cat_val_arr[i]
                sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=i#print(prop_sort_cat)
        elif method=='extend':
            """
            range of b values
                [o ......o......o      ...      o] 
              div1    div2    div3    ...     divN
            |...o...|...o...|...o...|  ...  |...o...|
            """
            cat_range_extend=(b_max-b_min)/(2*(num_cat-1))
            cat_range=b_range+2*cat_range_extend
            cat_min=b_min-cat_range_extend
            cat_max=b_max+cat_range_extend
            cat_width=(cat_max-cat_min)/num_cat
            cat_val_arr=np.linspace(cat_min+cat_width/2, cat_max-cat_width/2, num_cat)
            cat_range_arr=np.linspace(cat_min,cat_max,num_cat+1)    
            b_sort_cat=np.zeros(len(b_sort))
            sort_cat=np.zeros(len(b_sort))
            for i in range(len(cat_val_arr)):
                # lower and upper bound of current category in sorted b array
                cat_low_bound=cat_range_arr[i]
                cat_low_bound_arg=len(b_sort[b_sort<cat_low_bound])
                cat_upper_bound=cat_range_arr[i+1]
                cat_upper_bound_arg=len(b_sort[b_sort<=cat_upper_bound])
                # cut sorted b array between lower and upper bound of current category
                # reset categorical b array values to b value of category
                # reset category array values to current category
                b_sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=cat_val_arr[i]
                sort_cat[cat_low_bound_arg:cat_upper_bound_arg]=i#print(prop_sort_cat)
        # undo sorting of b values
        pseudo_b[b_arg_sort]=b_sort_cat
        cat=np.zeros(len(pseudo_b), dtype='int')
        cat[b_arg_sort]=sort_cat
        return pseudo_b, cat