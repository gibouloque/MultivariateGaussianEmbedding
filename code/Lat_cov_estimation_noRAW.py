import numpy as np
from scipy.stats import linregress
import numpy.random as npr
import numpy.linalg as npl

import code.image_conversions as imc
from code.Lat_cov_estimation import generate_DCT_trans, generate_permuted_cov_small,generate_graph_cholesky_structure_from_perm,convert_dict_to_npy


"""
MAIN Functions
"""

def estimate_4lat_noRAW(j_im,noise_model, dc_model, C_corr):
    var_im = estimate_variance_from_dc(noise_model['a'],noise_model['b'],j_im , model=dc_model)
    var_im[var_im < 10**-4] = 10**-4
    var_im = imc.rolling_row_scan(np.pad(var_im, ((8,24), (8,24)), mode='wrap'), 3, stride=1,clipped=False )
    h,w,d = var_im.shape

    C = np.zeros((h,w,d,d), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            diag_std = np.sqrt(np.diag(var_im[i,j]))
            C[i,j] = (diag_std @ C_corr @ diag_std)
    chol_C = generate_permuted_cov_small(C)

    L1,L2,L3,L4 = generate_graph_cholesky_structure_from_perm(chol_C)
    L1,L2,L3,L4 = convert_dict_to_npy(L1, L2, L3, L4)
    return(L1,L2,L3,L4)

def estimate_DCT_hetero_model(im,a,b,H0, max_v, rf=None):
    mu, noisy_im = generate_mu_noisy_from_im(im, a,b, max_v,rf)
    noise_dev, sigma_dev = generate_sigma_and_dev(mu, a,b,noisy_im,H0,rf)
    slope_est, intercept_est  = linreg_dc(noise_dev, sigma_dev)
    
    return(slope_est, intercept_est)

def generate_model_from_variance_map(var_im):
    a = np.zeros((33**2,64))

    k=0
    for i in np.arange(33):
        for j in np.arange(33):
            a[k] = zigzag(var_im[i*8:(i+1)*8,j*8:(j+1)*8])/var_im[i*8,j*8]
            k = k+1
    return(np.mean(a, axis=0))


    
def estimate_variance_from_dc(aDC, bDC, im, model):
    im_var = np.zeros_like(im)
    (h,w) = im.shape
    unsat_im = remove_saturated_zones(im)
   
    v = aDC*unsat_im+bDC
    v[unsat_im == 0] =10**-4
    for i in np.arange(h//8):
        for j in np.arange(w//8):
            im_var[8*i:(i+1)*8,8*j:(j+1)*8] = inv_zigzag_dct(v[8*i,8*j]* model).reshape(8,8)
    
    return(im_var)
"""
HELPER Functions
"""
def zigzag(a):
    return(np.concatenate([np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-a.shape[0], a.shape[0])]))
def zigzag_dct(a):
    return(a[ [0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5, 12,
       19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28, 35, 42,
       49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
       45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]])
def inv_zigzag_dct(a):
    return(a[ [0,  1,  5,  6, 14, 15, 27, 28,  2,  4,  7, 13, 16, 26, 29, 42,  3,
        8, 12, 17, 25, 30, 41, 43,  9, 11, 18, 24, 31, 40, 44, 53, 10, 19,
       23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37,
       47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]])
def estimate_cov_nosp(rb,a,b,spH,eps=10**-5,eps_max=1, lvl_max=2**16-1):
    spC_list = []
    for i in np.arange(rb.shape[0]):
        spC_list.append([])
        for j in np.arange(rb.shape[1]):
            v = a*rb[i,j] +b
            v[v < 0] = eps
            v[v > a*(95*lvl_max/100)+b] = eps_max
            spR= np.diag(v).astype(np.float32)
            spC = spH @ spR @ spH.T
            #spC = sparsify(spC.todense().astype(np.float32), alpha=0.99)
            spC_list[i].append(spC)
    return(spC_list)         
def generate_mu_noisy_from_im(im,a,b,max_v = None, rf = None,eps_max=1, mu=None):
    if mu is None:
        if max_v is None:
            if np.abs(b) > 2**12:
                max_v = 2**16
            elif np.abs(b) > 128:
                max_v = 2**14
            else:
                max_v = 2**8

        if rf is not None:
            mu = np.repeat(np.repeat(im,rf,axis=1),rf,axis=0)#
        else:
            mu = np.copy(im)
        mu = np.pad(mu,((1,1),(1,1)), mode='wrap')


        mu = (mu/mu.max()*max_v)

        if (b <0):
            mu[a*mu <=-b] = (-b+10**-4)/a 
    mu[mu > 95*max_v/100] = (eps_max-b)/a  
    sigma = a*mu+b
    sigma[sigma > a*(95*max_v/100)+b] = eps_max
    
    noisy_im = npr.normal(mu, np.sqrt(sigma))
    
    return(mu,noisy_im)
def generate_sigma_and_dev(mu, a,b, noisy_im,H0,rf=None, H_down=None):
    H_DCT = generate_DCT_trans(1)
    if rf is not None:
        rb = imc.block_row_scan(mu, rf, add_margin=True)
        C = estimate_cov_nosp(rb,a,b,H_DCT@H0,eps=10**-5,eps_max=1, lvl_max=2**16-1)

    else:
        rb = imc.block_row_scan(noisy_im, 1, add_margin=True)
        C = estimate_cov_nosp(rb,a,b,H_DCT@H0,eps=10**-5,eps_max=1, lvl_max=2**16-1)

    var_map = extract_var_map(C)
    h,w,d = var_map.shape

    sigma_dev = imc.unblock_row_scan(var_map, 264,264)
        
    if rf is not None:
        
        noise_dev = (imc.block_row_scan(noisy_im, k=rf, add_margin=True).reshape(h*w,676) @ H0.T -128) @ H_DCT.T
    else:
        noise_dev = (imc.block_row_scan(noisy_im, k=1, add_margin=True).reshape(h*w,100) @ H0.T -128)@ H_DCT.T
    noise_dev = imc.unblock_row_scan(noise_dev.reshape(h,w,d),264,264)
    
    return(noise_dev, sigma_dev)
def linreg_dc(noise_im, sigma_im):
    slope, intercept,_,_,_  = linregress(noise_im[::8,::8].ravel(),sigma_im[::8,::8].ravel())
    return(slope,intercept)


def remove_saturated_zones(im):
    (h,w) = im.shape
    im_spat = imc.compute_spatial_domain(im, np.ones((8,8)))
    mask_sat = im_spat >= 250
    maskb = imc.block_row_scan(mask_sat,1)
    for i in range(maskb.shape[0]):
        for j in range(maskb.shape[1]):
            if (maskb[i,j]).any():
                maskb[i,j,:] =True
    mask = imc.unblock_row_scan(maskb,h,w).astype(np.bool)
    im[mask] = 0
    return(im)



def extract_var_map(C):
    m,n,d = len(C), len(C[0]), C[0][0].shape[0]
    idx = np.diag_indices(d)
    v = np.zeros((m,n,d))
    for i in range(m):
        for j in range(n):
            v[i,j,:] = np.array(C[i][j][idx])
    return(v)
