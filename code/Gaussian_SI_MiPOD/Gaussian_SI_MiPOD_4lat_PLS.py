
from sys import path
path.append('..')
import numpy as np
from numpy import random as npr
import numpy.linalg as npl

from scipy.stats import norm
import code.image_conversions as imc



class Gaussian_SI_MIPOD_process_4lat_PLS:
    """
    This class represents MIPOD algorithm
    """

    def __init__(self, payload, table):
        """
        Store the cover path and the payload
        """
        self.payload = float(payload)
        self.table = table

    def do_process(self,precover,L1,L2,L3,L4):#, alphabet_size=2):
            Mq = self.payload * precover.size*np.log(2)
            (h,w) = precover.shape
            precover = precover.ravel()

            cover = np.round(precover)
            stego_signal_im,seq_eps,seq_mu = embed_four_lattice(L1,L2,L3,L4)

            
            signal_scaler = compute_scale_mat_stego_signal_approx(precover.ravel(),seq_mu.ravel(), seq_eps.ravel(),Mq,self.table)
            new_sig =  signal_scaler*stego_signal_im.ravel()
            
            stego = precover + new_sig



            return(cover.reshape(h,w), stego.reshape(h,w), signal_scaler)
       
    def do_process_inde(self,precover,chol_C):
                Mq = self.payload * precover.size*np.log(2)
                (h,w) = precover.shape
                precover = precover.ravel()

                cover = np.round(precover)
                stego_signal_im,seq_eps,seq_mu = embed_inde_block(chol_C)
                seq_eps[seq_eps < 10**-7] = 10**-7

                signal_scaler = compute_scale_mat_stego_signal_approx(precover.ravel(),seq_mu.ravel(), seq_eps.ravel(),Mq,self.table)
                new_sig =  signal_scaler*stego_signal_im.ravel()
                
                stego = precover + new_sig
                return(cover.reshape(h,w), stego.reshape(h,w), signal_scaler)
            
        
    def do_process_inde_nb1(self,precover,chol_C):
                Mq = self.payload * precover.size*np.log(2)
                (h,w) = precover.shape
                precover = precover.ravel()

                cover = np.round(precover)
                stego_signal_im,seq_eps,seq_mu = embed_inde_block_nb1(chol_C)
                seq_eps[seq_eps < 10**-7] = 10**-7

                signal_scaler = compute_scale_mat_stego_signal_approx(precover.ravel(),seq_mu.ravel(), seq_eps.ravel(),Mq,self.table)
                new_sig =  signal_scaler*stego_signal_im.ravel()
                
                stego = precover + new_sig
                return(cover.reshape(h,w), stego.reshape(h,w), signal_scaler)
def embed_inde_block(chol_C):
    d = chol_C.shape[2]
    nw,nh = chol_C.shape[0],chol_C.shape[1]
    inde_sample = npr.standard_normal(d*nh*nw).reshape(nh,nw,d,1)
    corr_sample = chol_C @ inde_sample
    eps =chol_C.T[np.diag_indices(d)].T
    mu =  corr_sample[:,:,:,0] - eps*inde_sample[:,:,:,0]
    
    y = imc.unblock_row_scan_nb(corr_sample[:,:,:,0], 3, 264,264)
    eps = imc.unblock_row_scan_nb(eps, 3,264,264)
    mu = imc.unblock_row_scan_nb(mu, 3,264,264)
    return(y,eps,mu)
def embed_inde_block_nb1(chol_C):
    d = chol_C.shape[2]
    nw,nh = chol_C.shape[0],chol_C.shape[1]
    inde_sample = npr.standard_normal(d*nh*nw).reshape(nh,nw,d,1)
    corr_sample = chol_C @ inde_sample
    eps =chol_C.T[np.diag_indices(d)].T
    mu =  corr_sample[:,:,:,0] - eps*inde_sample[:,:,:,0]
    
    y = imc.unblock_row_scan(corr_sample[:,:,:,0], 264,264)
    eps = imc.unblock_row_scan(eps,264,264)
    mu = imc.unblock_row_scan(mu, 264,264)
    return(y,eps,mu)
def compute_scale_mat_stego_signal_approx(precover, seq_mu,seq_eps, Mq, table):
    L = 0.1
    R = 1
    
    e = precover - np.round(precover)
    fL =  compute_discrete_entropy(e + L*seq_mu - np.round(e+L*seq_mu), L*seq_eps, table) - Mq
    
    fR =  compute_discrete_entropy(e + R*seq_mu - np.round(e+R*seq_mu), R*seq_eps,table) - Mq
    
    
    
    while fL*fR  > 0:
        if fL > 0:
            L = L/2
            fL =  compute_discrete_entropy(e + L*seq_mu - np.round(e+L*seq_mu), L*seq_eps,table) - Mq
        else:
            R = 2*R
            fR =  compute_discrete_entropy(e + R*seq_mu - np.round(e+R*seq_mu), R*seq_eps,table) - Mq
        #print(fL, fR)
            

    i=0
    maxiter = 500
    tolerance = 1
    fM =tolerance+1
    TM = np.zeros((maxiter,2))
    while (np.abs(fM)>tolerance) & (i<maxiter): 
        #print(fM)
        M = (L+R)/2;
        fM =  compute_discrete_entropy(e + M*seq_mu - np.round(e+M*seq_mu), M*seq_eps,table) - Mq
        #print(fM)
        if fL*fM < 0:
            R = M
            fR = fM
        else:
            L = M
            fL = fM
        
        TM[i,:] = [fM,M]
        i = i + 1
    if (i==maxiter):
        tmp_M = TM[np.abs(TM[:,0]) == np.min(np.abs(TM[:,0])),1]
        M = tmp_M[0]
    return(M)
def compute_discrete_entropy(mu, std, table):
    h = np.zeros(mu.size)
    h[std>0.5] = 1.4189385332046727+ 0.5*np.log(std[std>0.5]**2+1/12)
    
    idx_mu_small = (np.abs(np.around(mu[std<0.5], 3))*1000).astype(np.int16)
    idx_std_small = (np.around(std[std<0.5], 3)*1000).astype(np.int16)
    h[std<0.5] = table[idx_mu_small,idx_std_small]
    return(np.sum(h))
def embed_four_lattice(L1,L2,L3,L4, h=264,w=264):

    L_lookup = [[-1,-1],[-1,0],[-1,1],[0,-1],
                 [0,0],[0,1],[1,-1],[1,0],[1,1]]
    x = npr.standard_normal((h+48)*(w+48)).reshape(h+48,w+48)
    xb = imc.block_row_scan(x, 3)
    xb_view = imc.rolling_row_scan(x, 3, stride=1)
    y = np.zeros_like(xb)
    sigma_bar = np.zeros_like(xb)
    mu_bar = np.zeros_like(xb)

    idx_L2 = generate_block_idx(np.array([0,2,6,8,4]))
    idx_L3 = generate_block_idx(np.array([3,5,1,7,4]))
    idx_L4 = generate_block_idx(np.array([1,7,3,5,0,2,6,8,4]))

    for key in range(L1.shape[0]):
        k = L1_keys[key]
        m,n = int(k[0]//3)+1, int(k[1]//3)+1

        y[m, n, k[2]*64:(k[2]+1)*64] = L1[key] @ xb_view[k[0], k[1], k[2]*64:(k[2]+1)*64] 
        sigma_bar[m,n,k[2]*64:(k[2]+1)*64] = np.diag(L1[key][-64:, -64:])
    for key in range(L2.shape[0]):
        k = L2_keys[key]
        m,n = int(k[0]//3)+1, int(k[1]//3)+1
        y[m, n, k[2]*64:(k[2]+1)*64] = L2[key] @ xb_view[k[0]+L_lookup[k[2]][0], k[1]+L_lookup[k[2]][1], idx_L2]
        sigma_bar[m,n,k[2]*64:(k[2]+1)*64] = np.diag(L2[key][-64:, -64:])
    for key in range(L3.shape[0]):
        k = L3_keys[key]
        m,n = int(k[0]//3)+1, int(k[1]//3)+1
        y[m, n, k[2]*64:(k[2]+1)*64] = L3[key] @ xb_view[k[0]+L_lookup[k[2]][0], k[1]+L_lookup[k[2]][1], idx_L3] 
        sigma_bar[m,n,k[2]*64:(k[2]+1)*64] = np.diag(L3[key][-64:, -64:])
    for key in range(L4.shape[0]):
        k = L4_keys[key]
        m,n = int(k[0]//3)+1, int(k[1]//3)+1
        y[m, n, k[2]*64:(k[2]+1)*64] = L4[key] @ xb_view[k[0]+L_lookup[k[2]][0], k[1]+L_lookup[k[2]][1], idx_L4]
        sigma_bar[m,n,k[2]*64:(k[2]+1)*64] = np.diag(L4[key][-64:, -64:])
    y = imc.unblock_row_scan_nb(y[1:-1, 1:-1], 3, 264,264)
    sigma_bar = imc.unblock_row_scan_nb(sigma_bar[1:-1, 1:-1], 3, 264,264)
    mu_bar = y - sigma_bar*x[8:-40, 8:-40]
    return(y,sigma_bar,mu_bar)

def generate_block_idx(block_n_list):
    idx = []
    for i in np.arange(block_n_list.size):
        idx = np.concatenate([idx, np.arange(block_n_list[i]*64,(block_n_list[i]+1)*64)]).astype(np.int)
    return(idx)


def compute_Q_ary_entropy(beta_Z,beta_P,beta_M):
    SP = np.zeros_like(beta_P)
    SM = np.zeros_like(beta_M)
    SZ = np.zeros_like(beta_Z)
    for i in np.arange(beta_P.shape[1]):
        mask = beta_P[:,i] !=0
        t = beta_P[mask,i]
        SP[mask, i] = t*   np.log(t)
        mask = beta_M[:,i] !=0
        t = beta_M[mask,i]
        SM[mask, i] = t *   np.log(t)
    SP = np.nansum(SP, axis=1)
    SM = np.nansum(SM, axis=1)
    
    mask = beta_Z !=0
    t = beta_Z[mask]
    SZ[mask] = t*   np.log(t)
    
    return(-np.nansum(SP+SM + SZ))

def Q_ary_prob_seq_emb(cover, eps, eps_mu,alphabet_size):
    
    e = cover - np.round(cover)
    betaZ = np.zeros((cover.shape[0]))
    betaP=  np.zeros((cover.shape[0],alphabet_size))
    betaM = np.zeros((cover.shape[0],alphabet_size))
    betaZ = norm.cdf(0.5, loc= e +eps_mu, scale=eps)-norm.cdf(-0.5,loc=e+eps_mu, scale=eps)
    for j in np.arange(1,alphabet_size+1):
            betaP[:,j-1] = norm.cdf(j + 0.5, loc= e+eps_mu, scale=eps)-norm.cdf(j-0.5, loc=e+eps_mu, scale=eps)
            betaM[:,j-1] = norm.cdf(-j +0.5, loc= e+eps_mu, scale=eps)-norm.cdf(-j-0.5,loc=e+eps_mu, scale=eps)
    beta_sum = betaZ + np.sum(betaM, axis=1) +np.sum(betaP, axis=1)
    return(betaZ/beta_sum, betaP/beta_sum[:,None], betaM/beta_sum[:,None])


L1_keys = {0: (1, 1, 0),
 1: (1, 1, 2),
 2: (1, 1, 6),
 3: (1, 1, 8),
 4: (1, 7, 0),
 5: (1, 7, 2),
 6: (1, 7, 6),
 7: (1, 7, 8),
 8: (1, 13, 0),
 9: (1, 13, 2),
 10: (1, 13, 6),
 11: (1, 13, 8),
 12: (1, 19, 0),
 13: (1, 19, 2),
 14: (1, 19, 6),
 15: (1, 19, 8),
 16: (1, 25, 0),
 17: (1, 25, 2),
 18: (1, 25, 6),
 19: (1, 25, 8),
 20: (1, 31, 0),
 21: (1, 31, 2),
 22: (1, 31, 6),
 23: (1, 31, 8),
 24: (7, 1, 0),
 25: (7, 1, 2),
 26: (7, 1, 6),
 27: (7, 1, 8),
 28: (7, 7, 0),
 29: (7, 7, 2),
 30: (7, 7, 6),
 31: (7, 7, 8),
 32: (7, 13, 0),
 33: (7, 13, 2),
 34: (7, 13, 6),
 35: (7, 13, 8),
 36: (7, 19, 0),
 37: (7, 19, 2),
 38: (7, 19, 6),
 39: (7, 19, 8),
 40: (7, 25, 0),
 41: (7, 25, 2),
 42: (7, 25, 6),
 43: (7, 25, 8),
 44: (7, 31, 0),
 45: (7, 31, 2),
 46: (7, 31, 6),
 47: (7, 31, 8),
 48: (13, 1, 0),
 49: (13, 1, 2),
 50: (13, 1, 6),
 51: (13, 1, 8),
 52: (13, 7, 0),
 53: (13, 7, 2),
 54: (13, 7, 6),
 55: (13, 7, 8),
 56: (13, 13, 0),
 57: (13, 13, 2),
 58: (13, 13, 6),
 59: (13, 13, 8),
 60: (13, 19, 0),
 61: (13, 19, 2),
 62: (13, 19, 6),
 63: (13, 19, 8),
 64: (13, 25, 0),
 65: (13, 25, 2),
 66: (13, 25, 6),
 67: (13, 25, 8),
 68: (13, 31, 0),
 69: (13, 31, 2),
 70: (13, 31, 6),
 71: (13, 31, 8),
 72: (19, 1, 0),
 73: (19, 1, 2),
 74: (19, 1, 6),
 75: (19, 1, 8),
 76: (19, 7, 0),
 77: (19, 7, 2),
 78: (19, 7, 6),
 79: (19, 7, 8),
 80: (19, 13, 0),
 81: (19, 13, 2),
 82: (19, 13, 6),
 83: (19, 13, 8),
 84: (19, 19, 0),
 85: (19, 19, 2),
 86: (19, 19, 6),
 87: (19, 19, 8),
 88: (19, 25, 0),
 89: (19, 25, 2),
 90: (19, 25, 6),
 91: (19, 25, 8),
 92: (19, 31, 0),
 93: (19, 31, 2),
 94: (19, 31, 6),
 95: (19, 31, 8),
 96: (25, 1, 0),
 97: (25, 1, 2),
 98: (25, 1, 6),
 99: (25, 1, 8),
 100: (25, 7, 0),
 101: (25, 7, 2),
 102: (25, 7, 6),
 103: (25, 7, 8),
 104: (25, 13, 0),
 105: (25, 13, 2),
 106: (25, 13, 6),
 107: (25, 13, 8),
 108: (25, 19, 0),
 109: (25, 19, 2),
 110: (25, 19, 6),
 111: (25, 19, 8),
 112: (25, 25, 0),
 113: (25, 25, 2),
 114: (25, 25, 6),
 115: (25, 25, 8),
 116: (25, 31, 0),
 117: (25, 31, 2),
 118: (25, 31, 6),
 119: (25, 31, 8),
 120: (31, 1, 0),
 121: (31, 1, 2),
 122: (31, 1, 6),
 123: (31, 1, 8),
 124: (31, 7, 0),
 125: (31, 7, 2),
 126: (31, 7, 6),
 127: (31, 7, 8),
 128: (31, 13, 0),
 129: (31, 13, 2),
 130: (31, 13, 6),
 131: (31, 13, 8),
 132: (31, 19, 0),
 133: (31, 19, 2),
 134: (31, 19, 6),
 135: (31, 19, 8),
 136: (31, 25, 0),
 137: (31, 25, 2),
 138: (31, 25, 6),
 139: (31, 25, 8),
 140: (31, 31, 0),
 141: (31, 31, 2),
 142: (31, 31, 6),
 143: (31, 31, 8),
 144: (1, 4, 1),
 145: (1, 4, 7),
 146: (1, 10, 1),
 147: (1, 10, 7),
 148: (1, 16, 1),
 149: (1, 16, 7),
 150: (1, 22, 1),
 151: (1, 22, 7),
 152: (1, 28, 1),
 153: (1, 28, 7),
 154: (7, 4, 1),
 155: (7, 4, 7),
 156: (7, 10, 1),
 157: (7, 10, 7),
 158: (7, 16, 1),
 159: (7, 16, 7),
 160: (7, 22, 1),
 161: (7, 22, 7),
 162: (7, 28, 1),
 163: (7, 28, 7),
 164: (13, 4, 1),
 165: (13, 4, 7),
 166: (13, 10, 1),
 167: (13, 10, 7),
 168: (13, 16, 1),
 169: (13, 16, 7),
 170: (13, 22, 1),
 171: (13, 22, 7),
 172: (13, 28, 1),
 173: (13, 28, 7),
 174: (19, 4, 1),
 175: (19, 4, 7),
 176: (19, 10, 1),
 177: (19, 10, 7),
 178: (19, 16, 1),
 179: (19, 16, 7),
 180: (19, 22, 1),
 181: (19, 22, 7),
 182: (19, 28, 1),
 183: (19, 28, 7),
 184: (25, 4, 1),
 185: (25, 4, 7),
 186: (25, 10, 1),
 187: (25, 10, 7),
 188: (25, 16, 1),
 189: (25, 16, 7),
 190: (25, 22, 1),
 191: (25, 22, 7),
 192: (25, 28, 1),
 193: (25, 28, 7),
 194: (31, 4, 1),
 195: (31, 4, 7),
 196: (31, 10, 1),
 197: (31, 10, 7),
 198: (31, 16, 1),
 199: (31, 16, 7),
 200: (31, 22, 1),
 201: (31, 22, 7),
 202: (31, 28, 1),
 203: (31, 28, 7),
 204: (4, 1, 3),
 205: (4, 1, 5),
 206: (4, 7, 3),
 207: (4, 7, 5),
 208: (4, 13, 3),
 209: (4, 13, 5),
 210: (4, 19, 3),
 211: (4, 19, 5),
 212: (4, 25, 3),
 213: (4, 25, 5),
 214: (4, 31, 3),
 215: (4, 31, 5),
 216: (10, 1, 3),
 217: (10, 1, 5),
 218: (10, 7, 3),
 219: (10, 7, 5),
 220: (10, 13, 3),
 221: (10, 13, 5),
 222: (10, 19, 3),
 223: (10, 19, 5),
 224: (10, 25, 3),
 225: (10, 25, 5),
 226: (10, 31, 3),
 227: (10, 31, 5),
 228: (16, 1, 3),
 229: (16, 1, 5),
 230: (16, 7, 3),
 231: (16, 7, 5),
 232: (16, 13, 3),
 233: (16, 13, 5),
 234: (16, 19, 3),
 235: (16, 19, 5),
 236: (16, 25, 3),
 237: (16, 25, 5),
 238: (16, 31, 3),
 239: (16, 31, 5),
 240: (22, 1, 3),
 241: (22, 1, 5),
 242: (22, 7, 3),
 243: (22, 7, 5),
 244: (22, 13, 3),
 245: (22, 13, 5),
 246: (22, 19, 3),
 247: (22, 19, 5),
 248: (22, 25, 3),
 249: (22, 25, 5),
 250: (22, 31, 3),
 251: (22, 31, 5),
 252: (28, 1, 3),
 253: (28, 1, 5),
 254: (28, 7, 3),
 255: (28, 7, 5),
 256: (28, 13, 3),
 257: (28, 13, 5),
 258: (28, 19, 3),
 259: (28, 19, 5),
 260: (28, 25, 3),
 261: (28, 25, 5),
 262: (28, 31, 3),
 263: (28, 31, 5),
 264: (4, 4, 4),
 265: (4, 10, 4),
 266: (4, 16, 4),
 267: (4, 22, 4),
 268: (4, 28, 4),
 269: (10, 4, 4),
 270: (10, 10, 4),
 271: (10, 16, 4),
 272: (10, 22, 4),
 273: (10, 28, 4),
 274: (16, 4, 4),
 275: (16, 10, 4),
 276: (16, 16, 4),
 277: (16, 22, 4),
 278: (16, 28, 4),
 279: (22, 4, 4),
 280: (22, 10, 4),
 281: (22, 16, 4),
 282: (22, 22, 4),
 283: (22, 28, 4),
 284: (28, 4, 4),
 285: (28, 10, 4),
 286: (28, 16, 4),
 287: (28, 22, 4),
 288: (28, 28, 4)}

L2_keys = {0: (1, 1, 4),
 1: (1, 7, 4),
 2: (1, 13, 4),
 3: (1, 19, 4),
 4: (1, 25, 4),
 5: (1, 31, 4),
 6: (7, 1, 4),
 7: (7, 7, 4),
 8: (7, 13, 4),
 9: (7, 19, 4),
 10: (7, 25, 4),
 11: (7, 31, 4),
 12: (13, 1, 4),
 13: (13, 7, 4),
 14: (13, 13, 4),
 15: (13, 19, 4),
 16: (13, 25, 4),
 17: (13, 31, 4),
 18: (19, 1, 4),
 19: (19, 7, 4),
 20: (19, 13, 4),
 21: (19, 19, 4),
 22: (19, 25, 4),
 23: (19, 31, 4),
 24: (25, 1, 4),
 25: (25, 7, 4),
 26: (25, 13, 4),
 27: (25, 19, 4),
 28: (25, 25, 4),
 29: (25, 31, 4),
 30: (31, 1, 4),
 31: (31, 7, 4),
 32: (31, 13, 4),
 33: (31, 19, 4),
 34: (31, 25, 4),
 35: (31, 31, 4),
 36: (1, 4, 3),
 37: (1, 4, 5),
 38: (1, 10, 3),
 39: (1, 10, 5),
 40: (1, 16, 3),
 41: (1, 16, 5),
 42: (1, 22, 3),
 43: (1, 22, 5),
 44: (1, 28, 3),
 45: (1, 28, 5),
 46: (7, 4, 3),
 47: (7, 4, 5),
 48: (7, 10, 3),
 49: (7, 10, 5),
 50: (7, 16, 3),
 51: (7, 16, 5),
 52: (7, 22, 3),
 53: (7, 22, 5),
 54: (7, 28, 3),
 55: (7, 28, 5),
 56: (13, 4, 3),
 57: (13, 4, 5),
 58: (13, 10, 3),
 59: (13, 10, 5),
 60: (13, 16, 3),
 61: (13, 16, 5),
 62: (13, 22, 3),
 63: (13, 22, 5),
 64: (13, 28, 3),
 65: (13, 28, 5),
 66: (19, 4, 3),
 67: (19, 4, 5),
 68: (19, 10, 3),
 69: (19, 10, 5),
 70: (19, 16, 3),
 71: (19, 16, 5),
 72: (19, 22, 3),
 73: (19, 22, 5),
 74: (19, 28, 3),
 75: (19, 28, 5),
 76: (25, 4, 3),
 77: (25, 4, 5),
 78: (25, 10, 3),
 79: (25, 10, 5),
 80: (25, 16, 3),
 81: (25, 16, 5),
 82: (25, 22, 3),
 83: (25, 22, 5),
 84: (25, 28, 3),
 85: (25, 28, 5),
 86: (31, 4, 3),
 87: (31, 4, 5),
 88: (31, 10, 3),
 89: (31, 10, 5),
 90: (31, 16, 3),
 91: (31, 16, 5),
 92: (31, 22, 3),
 93: (31, 22, 5),
 94: (31, 28, 3),
 95: (31, 28, 5),
 96: (4, 1, 1),
 97: (4, 1, 7),
 98: (4, 7, 1),
 99: (4, 7, 7),
 100: (4, 13, 1),
 101: (4, 13, 7),
 102: (4, 19, 1),
 103: (4, 19, 7),
 104: (4, 25, 1),
 105: (4, 25, 7),
 106: (4, 31, 1),
 107: (4, 31, 7),
 108: (10, 1, 1),
 109: (10, 1, 7),
 110: (10, 7, 1),
 111: (10, 7, 7),
 112: (10, 13, 1),
 113: (10, 13, 7),
 114: (10, 19, 1),
 115: (10, 19, 7),
 116: (10, 25, 1),
 117: (10, 25, 7),
 118: (10, 31, 1),
 119: (10, 31, 7),
 120: (16, 1, 1),
 121: (16, 1, 7),
 122: (16, 7, 1),
 123: (16, 7, 7),
 124: (16, 13, 1),
 125: (16, 13, 7),
 126: (16, 19, 1),
 127: (16, 19, 7),
 128: (16, 25, 1),
 129: (16, 25, 7),
 130: (16, 31, 1),
 131: (16, 31, 7),
 132: (22, 1, 1),
 133: (22, 1, 7),
 134: (22, 7, 1),
 135: (22, 7, 7),
 136: (22, 13, 1),
 137: (22, 13, 7),
 138: (22, 19, 1),
 139: (22, 19, 7),
 140: (22, 25, 1),
 141: (22, 25, 7),
 142: (22, 31, 1),
 143: (22, 31, 7),
 144: (28, 1, 1),
 145: (28, 1, 7),
 146: (28, 7, 1),
 147: (28, 7, 7),
 148: (28, 13, 1),
 149: (28, 13, 7),
 150: (28, 19, 1),
 151: (28, 19, 7),
 152: (28, 25, 1),
 153: (28, 25, 7),
 154: (28, 31, 1),
 155: (28, 31, 7),
 156: (4, 4, 0),
 157: (4, 4, 2),
 158: (4, 4, 6),
 159: (4, 4, 8),
 160: (4, 10, 0),
 161: (4, 10, 2),
 162: (4, 10, 6),
 163: (4, 10, 8),
 164: (4, 16, 0),
 165: (4, 16, 2),
 166: (4, 16, 6),
 167: (4, 16, 8),
 168: (4, 22, 0),
 169: (4, 22, 2),
 170: (4, 22, 6),
 171: (4, 22, 8),
 172: (4, 28, 0),
 173: (4, 28, 2),
 174: (4, 28, 6),
 175: (4, 28, 8),
 176: (10, 4, 0),
 177: (10, 4, 2),
 178: (10, 4, 6),
 179: (10, 4, 8),
 180: (10, 10, 0),
 181: (10, 10, 2),
 182: (10, 10, 6),
 183: (10, 10, 8),
 184: (10, 16, 0),
 185: (10, 16, 2),
 186: (10, 16, 6),
 187: (10, 16, 8),
 188: (10, 22, 0),
 189: (10, 22, 2),
 190: (10, 22, 6),
 191: (10, 22, 8),
 192: (10, 28, 0),
 193: (10, 28, 2),
 194: (10, 28, 6),
 195: (10, 28, 8),
 196: (16, 4, 0),
 197: (16, 4, 2),
 198: (16, 4, 6),
 199: (16, 4, 8),
 200: (16, 10, 0),
 201: (16, 10, 2),
 202: (16, 10, 6),
 203: (16, 10, 8),
 204: (16, 16, 0),
 205: (16, 16, 2),
 206: (16, 16, 6),
 207: (16, 16, 8),
 208: (16, 22, 0),
 209: (16, 22, 2),
 210: (16, 22, 6),
 211: (16, 22, 8),
 212: (16, 28, 0),
 213: (16, 28, 2),
 214: (16, 28, 6),
 215: (16, 28, 8),
 216: (22, 4, 0),
 217: (22, 4, 2),
 218: (22, 4, 6),
 219: (22, 4, 8),
 220: (22, 10, 0),
 221: (22, 10, 2),
 222: (22, 10, 6),
 223: (22, 10, 8),
 224: (22, 16, 0),
 225: (22, 16, 2),
 226: (22, 16, 6),
 227: (22, 16, 8),
 228: (22, 22, 0),
 229: (22, 22, 2),
 230: (22, 22, 6),
 231: (22, 22, 8),
 232: (22, 28, 0),
 233: (22, 28, 2),
 234: (22, 28, 6),
 235: (22, 28, 8),
 236: (28, 4, 0),
 237: (28, 4, 2),
 238: (28, 4, 6),
 239: (28, 4, 8),
 240: (28, 10, 0),
 241: (28, 10, 2),
 242: (28, 10, 6),
 243: (28, 10, 8),
 244: (28, 16, 0),
 245: (28, 16, 2),
 246: (28, 16, 6),
 247: (28, 16, 8),
 248: (28, 22, 0),
 249: (28, 22, 2),
 250: (28, 22, 6),
 251: (28, 22, 8),
 252: (28, 28, 0),
 253: (28, 28, 2),
 254: (28, 28, 6),
 255: (28, 28, 8)}

L3_keys = {0: (1, 1, 1),
 1: (1, 1, 7),
 2: (1, 7, 1),
 3: (1, 7, 7),
 4: (1, 13, 1),
 5: (1, 13, 7),
 6: (1, 19, 1),
 7: (1, 19, 7),
 8: (1, 25, 1),
 9: (1, 25, 7),
 10: (1, 31, 1),
 11: (1, 31, 7),
 12: (7, 1, 1),
 13: (7, 1, 7),
 14: (7, 7, 1),
 15: (7, 7, 7),
 16: (7, 13, 1),
 17: (7, 13, 7),
 18: (7, 19, 1),
 19: (7, 19, 7),
 20: (7, 25, 1),
 21: (7, 25, 7),
 22: (7, 31, 1),
 23: (7, 31, 7),
 24: (13, 1, 1),
 25: (13, 1, 7),
 26: (13, 7, 1),
 27: (13, 7, 7),
 28: (13, 13, 1),
 29: (13, 13, 7),
 30: (13, 19, 1),
 31: (13, 19, 7),
 32: (13, 25, 1),
 33: (13, 25, 7),
 34: (13, 31, 1),
 35: (13, 31, 7),
 36: (19, 1, 1),
 37: (19, 1, 7),
 38: (19, 7, 1),
 39: (19, 7, 7),
 40: (19, 13, 1),
 41: (19, 13, 7),
 42: (19, 19, 1),
 43: (19, 19, 7),
 44: (19, 25, 1),
 45: (19, 25, 7),
 46: (19, 31, 1),
 47: (19, 31, 7),
 48: (25, 1, 1),
 49: (25, 1, 7),
 50: (25, 7, 1),
 51: (25, 7, 7),
 52: (25, 13, 1),
 53: (25, 13, 7),
 54: (25, 19, 1),
 55: (25, 19, 7),
 56: (25, 25, 1),
 57: (25, 25, 7),
 58: (25, 31, 1),
 59: (25, 31, 7),
 60: (31, 1, 1),
 61: (31, 1, 7),
 62: (31, 7, 1),
 63: (31, 7, 7),
 64: (31, 13, 1),
 65: (31, 13, 7),
 66: (31, 19, 1),
 67: (31, 19, 7),
 68: (31, 25, 1),
 69: (31, 25, 7),
 70: (31, 31, 1),
 71: (31, 31, 7),
 72: (1, 4, 0),
 73: (1, 4, 2),
 74: (1, 4, 6),
 75: (1, 4, 8),
 76: (1, 10, 0),
 77: (1, 10, 2),
 78: (1, 10, 6),
 79: (1, 10, 8),
 80: (1, 16, 0),
 81: (1, 16, 2),
 82: (1, 16, 6),
 83: (1, 16, 8),
 84: (1, 22, 0),
 85: (1, 22, 2),
 86: (1, 22, 6),
 87: (1, 22, 8),
 88: (1, 28, 0),
 89: (1, 28, 2),
 90: (1, 28, 6),
 91: (1, 28, 8),
 92: (7, 4, 0),
 93: (7, 4, 2),
 94: (7, 4, 6),
 95: (7, 4, 8),
 96: (7, 10, 0),
 97: (7, 10, 2),
 98: (7, 10, 6),
 99: (7, 10, 8),
 100: (7, 16, 0),
 101: (7, 16, 2),
 102: (7, 16, 6),
 103: (7, 16, 8),
 104: (7, 22, 0),
 105: (7, 22, 2),
 106: (7, 22, 6),
 107: (7, 22, 8),
 108: (7, 28, 0),
 109: (7, 28, 2),
 110: (7, 28, 6),
 111: (7, 28, 8),
 112: (13, 4, 0),
 113: (13, 4, 2),
 114: (13, 4, 6),
 115: (13, 4, 8),
 116: (13, 10, 0),
 117: (13, 10, 2),
 118: (13, 10, 6),
 119: (13, 10, 8),
 120: (13, 16, 0),
 121: (13, 16, 2),
 122: (13, 16, 6),
 123: (13, 16, 8),
 124: (13, 22, 0),
 125: (13, 22, 2),
 126: (13, 22, 6),
 127: (13, 22, 8),
 128: (13, 28, 0),
 129: (13, 28, 2),
 130: (13, 28, 6),
 131: (13, 28, 8),
 132: (19, 4, 0),
 133: (19, 4, 2),
 134: (19, 4, 6),
 135: (19, 4, 8),
 136: (19, 10, 0),
 137: (19, 10, 2),
 138: (19, 10, 6),
 139: (19, 10, 8),
 140: (19, 16, 0),
 141: (19, 16, 2),
 142: (19, 16, 6),
 143: (19, 16, 8),
 144: (19, 22, 0),
 145: (19, 22, 2),
 146: (19, 22, 6),
 147: (19, 22, 8),
 148: (19, 28, 0),
 149: (19, 28, 2),
 150: (19, 28, 6),
 151: (19, 28, 8),
 152: (25, 4, 0),
 153: (25, 4, 2),
 154: (25, 4, 6),
 155: (25, 4, 8),
 156: (25, 10, 0),
 157: (25, 10, 2),
 158: (25, 10, 6),
 159: (25, 10, 8),
 160: (25, 16, 0),
 161: (25, 16, 2),
 162: (25, 16, 6),
 163: (25, 16, 8),
 164: (25, 22, 0),
 165: (25, 22, 2),
 166: (25, 22, 6),
 167: (25, 22, 8),
 168: (25, 28, 0),
 169: (25, 28, 2),
 170: (25, 28, 6),
 171: (25, 28, 8),
 172: (31, 4, 0),
 173: (31, 4, 2),
 174: (31, 4, 6),
 175: (31, 4, 8),
 176: (31, 10, 0),
 177: (31, 10, 2),
 178: (31, 10, 6),
 179: (31, 10, 8),
 180: (31, 16, 0),
 181: (31, 16, 2),
 182: (31, 16, 6),
 183: (31, 16, 8),
 184: (31, 22, 0),
 185: (31, 22, 2),
 186: (31, 22, 6),
 187: (31, 22, 8),
 188: (31, 28, 0),
 189: (31, 28, 2),
 190: (31, 28, 6),
 191: (31, 28, 8),
 192: (4, 1, 4),
 193: (4, 7, 4),
 194: (4, 13, 4),
 195: (4, 19, 4),
 196: (4, 25, 4),
 197: (4, 31, 4),
 198: (10, 1, 4),
 199: (10, 7, 4),
 200: (10, 13, 4),
 201: (10, 19, 4),
 202: (10, 25, 4),
 203: (10, 31, 4),
 204: (16, 1, 4),
 205: (16, 7, 4),
 206: (16, 13, 4),
 207: (16, 19, 4),
 208: (16, 25, 4),
 209: (16, 31, 4),
 210: (22, 1, 4),
 211: (22, 7, 4),
 212: (22, 13, 4),
 213: (22, 19, 4),
 214: (22, 25, 4),
 215: (22, 31, 4),
 216: (28, 1, 4),
 217: (28, 7, 4),
 218: (28, 13, 4),
 219: (28, 19, 4),
 220: (28, 25, 4),
 221: (28, 31, 4),
 222: (4, 4, 3),
 223: (4, 4, 5),
 224: (4, 10, 3),
 225: (4, 10, 5),
 226: (4, 16, 3),
 227: (4, 16, 5),
 228: (4, 22, 3),
 229: (4, 22, 5),
 230: (4, 28, 3),
 231: (4, 28, 5),
 232: (10, 4, 3),
 233: (10, 4, 5),
 234: (10, 10, 3),
 235: (10, 10, 5),
 236: (10, 16, 3),
 237: (10, 16, 5),
 238: (10, 22, 3),
 239: (10, 22, 5),
 240: (10, 28, 3),
 241: (10, 28, 5),
 242: (16, 4, 3),
 243: (16, 4, 5),
 244: (16, 10, 3),
 245: (16, 10, 5),
 246: (16, 16, 3),
 247: (16, 16, 5),
 248: (16, 22, 3),
 249: (16, 22, 5),
 250: (16, 28, 3),
 251: (16, 28, 5),
 252: (22, 4, 3),
 253: (22, 4, 5),
 254: (22, 10, 3),
 255: (22, 10, 5),
 256: (22, 16, 3),
 257: (22, 16, 5),
 258: (22, 22, 3),
 259: (22, 22, 5),
 260: (22, 28, 3),
 261: (22, 28, 5),
 262: (28, 4, 3),
 263: (28, 4, 5),
 264: (28, 10, 3),
 265: (28, 10, 5),
 266: (28, 16, 3),
 267: (28, 16, 5),
 268: (28, 22, 3),
 269: (28, 22, 5),
 270: (28, 28, 3),
 271: (28, 28, 5)}

L4_keys = {0: (1, 1, 5),
 1: (1, 1, 3),
 2: (1, 7, 5),
 3: (1, 7, 3),
 4: (1, 13, 5),
 5: (1, 13, 3),
 6: (1, 19, 5),
 7: (1, 19, 3),
 8: (1, 25, 5),
 9: (1, 25, 3),
 10: (1, 31, 5),
 11: (1, 31, 3),
 12: (7, 1, 5),
 13: (7, 1, 3),
 14: (7, 7, 5),
 15: (7, 7, 3),
 16: (7, 13, 5),
 17: (7, 13, 3),
 18: (7, 19, 5),
 19: (7, 19, 3),
 20: (7, 25, 5),
 21: (7, 25, 3),
 22: (7, 31, 5),
 23: (7, 31, 3),
 24: (13, 1, 5),
 25: (13, 1, 3),
 26: (13, 7, 5),
 27: (13, 7, 3),
 28: (13, 13, 5),
 29: (13, 13, 3),
 30: (13, 19, 5),
 31: (13, 19, 3),
 32: (13, 25, 5),
 33: (13, 25, 3),
 34: (13, 31, 5),
 35: (13, 31, 3),
 36: (19, 1, 5),
 37: (19, 1, 3),
 38: (19, 7, 5),
 39: (19, 7, 3),
 40: (19, 13, 5),
 41: (19, 13, 3),
 42: (19, 19, 5),
 43: (19, 19, 3),
 44: (19, 25, 5),
 45: (19, 25, 3),
 46: (19, 31, 5),
 47: (19, 31, 3),
 48: (25, 1, 5),
 49: (25, 1, 3),
 50: (25, 7, 5),
 51: (25, 7, 3),
 52: (25, 13, 5),
 53: (25, 13, 3),
 54: (25, 19, 5),
 55: (25, 19, 3),
 56: (25, 25, 5),
 57: (25, 25, 3),
 58: (25, 31, 5),
 59: (25, 31, 3),
 60: (31, 1, 5),
 61: (31, 1, 3),
 62: (31, 7, 5),
 63: (31, 7, 3),
 64: (31, 13, 5),
 65: (31, 13, 3),
 66: (31, 19, 5),
 67: (31, 19, 3),
 68: (31, 25, 5),
 69: (31, 25, 3),
 70: (31, 31, 5),
 71: (31, 31, 3),
 72: (1, 4, 4),
 73: (1, 10, 4),
 74: (1, 16, 4),
 75: (1, 22, 4),
 76: (1, 28, 4),
 77: (7, 4, 4),
 78: (7, 10, 4),
 79: (7, 16, 4),
 80: (7, 22, 4),
 81: (7, 28, 4),
 82: (13, 4, 4),
 83: (13, 10, 4),
 84: (13, 16, 4),
 85: (13, 22, 4),
 86: (13, 28, 4),
 87: (19, 4, 4),
 88: (19, 10, 4),
 89: (19, 16, 4),
 90: (19, 22, 4),
 91: (19, 28, 4),
 92: (25, 4, 4),
 93: (25, 10, 4),
 94: (25, 16, 4),
 95: (25, 22, 4),
 96: (25, 28, 4),
 97: (31, 4, 4),
 98: (31, 10, 4),
 99: (31, 16, 4),
 100: (31, 22, 4),
 101: (31, 28, 4),
 102: (4, 1, 0),
 103: (4, 1, 2),
 104: (4, 1, 6),
 105: (4, 1, 8),
 106: (4, 7, 0),
 107: (4, 7, 2),
 108: (4, 7, 6),
 109: (4, 7, 8),
 110: (4, 13, 0),
 111: (4, 13, 2),
 112: (4, 13, 6),
 113: (4, 13, 8),
 114: (4, 19, 0),
 115: (4, 19, 2),
 116: (4, 19, 6),
 117: (4, 19, 8),
 118: (4, 25, 0),
 119: (4, 25, 2),
 120: (4, 25, 6),
 121: (4, 25, 8),
 122: (4, 31, 0),
 123: (4, 31, 2),
 124: (4, 31, 6),
 125: (4, 31, 8),
 126: (10, 1, 0),
 127: (10, 1, 2),
 128: (10, 1, 6),
 129: (10, 1, 8),
 130: (10, 7, 0),
 131: (10, 7, 2),
 132: (10, 7, 6),
 133: (10, 7, 8),
 134: (10, 13, 0),
 135: (10, 13, 2),
 136: (10, 13, 6),
 137: (10, 13, 8),
 138: (10, 19, 0),
 139: (10, 19, 2),
 140: (10, 19, 6),
 141: (10, 19, 8),
 142: (10, 25, 0),
 143: (10, 25, 2),
 144: (10, 25, 6),
 145: (10, 25, 8),
 146: (10, 31, 0),
 147: (10, 31, 2),
 148: (10, 31, 6),
 149: (10, 31, 8),
 150: (16, 1, 0),
 151: (16, 1, 2),
 152: (16, 1, 6),
 153: (16, 1, 8),
 154: (16, 7, 0),
 155: (16, 7, 2),
 156: (16, 7, 6),
 157: (16, 7, 8),
 158: (16, 13, 0),
 159: (16, 13, 2),
 160: (16, 13, 6),
 161: (16, 13, 8),
 162: (16, 19, 0),
 163: (16, 19, 2),
 164: (16, 19, 6),
 165: (16, 19, 8),
 166: (16, 25, 0),
 167: (16, 25, 2),
 168: (16, 25, 6),
 169: (16, 25, 8),
 170: (16, 31, 0),
 171: (16, 31, 2),
 172: (16, 31, 6),
 173: (16, 31, 8),
 174: (22, 1, 0),
 175: (22, 1, 2),
 176: (22, 1, 6),
 177: (22, 1, 8),
 178: (22, 7, 0),
 179: (22, 7, 2),
 180: (22, 7, 6),
 181: (22, 7, 8),
 182: (22, 13, 0),
 183: (22, 13, 2),
 184: (22, 13, 6),
 185: (22, 13, 8),
 186: (22, 19, 0),
 187: (22, 19, 2),
 188: (22, 19, 6),
 189: (22, 19, 8),
 190: (22, 25, 0),
 191: (22, 25, 2),
 192: (22, 25, 6),
 193: (22, 25, 8),
 194: (22, 31, 0),
 195: (22, 31, 2),
 196: (22, 31, 6),
 197: (22, 31, 8),
 198: (28, 1, 0),
 199: (28, 1, 2),
 200: (28, 1, 6),
 201: (28, 1, 8),
 202: (28, 7, 0),
 203: (28, 7, 2),
 204: (28, 7, 6),
 205: (28, 7, 8),
 206: (28, 13, 0),
 207: (28, 13, 2),
 208: (28, 13, 6),
 209: (28, 13, 8),
 210: (28, 19, 0),
 211: (28, 19, 2),
 212: (28, 19, 6),
 213: (28, 19, 8),
 214: (28, 25, 0),
 215: (28, 25, 2),
 216: (28, 25, 6),
 217: (28, 25, 8),
 218: (28, 31, 0),
 219: (28, 31, 2),
 220: (28, 31, 6),
 221: (28, 31, 8),
 222: (4, 4, 1),
 223: (4, 4, 7),
 224: (4, 10, 1),
 225: (4, 10, 7),
 226: (4, 16, 1),
 227: (4, 16, 7),
 228: (4, 22, 1),
 229: (4, 22, 7),
 230: (4, 28, 1),
 231: (4, 28, 7),
 232: (10, 4, 1),
 233: (10, 4, 7),
 234: (10, 10, 1),
 235: (10, 10, 7),
 236: (10, 16, 1),
 237: (10, 16, 7),
 238: (10, 22, 1),
 239: (10, 22, 7),
 240: (10, 28, 1),
 241: (10, 28, 7),
 242: (16, 4, 1),
 243: (16, 4, 7),
 244: (16, 10, 1),
 245: (16, 10, 7),
 246: (16, 16, 1),
 247: (16, 16, 7),
 248: (16, 22, 1),
 249: (16, 22, 7),
 250: (16, 28, 1),
 251: (16, 28, 7),
 252: (22, 4, 1),
 253: (22, 4, 7),
 254: (22, 10, 1),
 255: (22, 10, 7),
 256: (22, 16, 1),
 257: (22, 16, 7),
 258: (22, 22, 1),
 259: (22, 22, 7),
 260: (22, 28, 1),
 261: (22, 28, 7),
 262: (28, 4, 1),
 263: (28, 4, 7),
 264: (28, 10, 1),
 265: (28, 10, 7),
 266: (28, 16, 1),
 267: (28, 16, 7),
 268: (28, 22, 1),
 269: (28, 22, 7),
 270: (28, 28, 1),
 271: (28, 28, 7)}