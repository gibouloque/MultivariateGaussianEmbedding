import numpy as np
from .Gaussian_SI_MiPOD.Gaussian_SI_MiPOD_4lat_PLS import L1_keys, L2_keys, L3_keys, L4_keys
import numpy.random as npr
import numpy.linalg as npl
import scipy.sparse as sp
import code.image_conversions as imc

"""
MAIN FUNCTION
raw_block_view : Block view of the RAW file of the image, see notebook for examples
a,b : Heteroscedastic paramaters of the noise in the RAW domain
spH : Sparse matrix representation of the processing pipeline excluding the DCT transform
H_DCT : Matrix representation of the DCT transform -- use the generate_DCT_trans() helper function
lvl_max = Saturation level of the RAW image (sensor and ISO dependent)

"""

def estimate_4lat(raw_block_view,a,b,spH,H_DCT,lvl_max):
    spC_list = np.array(estimate_cov(raw_block_view,a,b,spH,lvl_max=lvl_max))
    spC_list = quant_C(spC_list, H_DCT)
    chol_C = generate_permuted_cov_small(spC_list)
    sPC_list = None # Free memory
    L1,L2,L3,L4 = generate_graph_cholesky_structure_from_perm(chol_C)
    L1,L2,L3,L4 = convert_dict_to_npy(L1, L2, L3, L4)
    return(L1,L2,L3,L4)



"""
HELPER FUNCTIONS
"""

def generate_DCT_trans(n_b, c_quant=np.ones((8,8))):
    n_c = n_b*8+2
    n_c_2 = n_b*8
    N = 8
    pi = np.pi
    C = np.sqrt(2.0/N)
    a, b, c, d, e, f, g = C*np.cos(pi/4) , C*np.cos(pi/16), C*np.cos(pi/8), C*np.cos(3*pi/16), C*np.cos(5*pi/16), C*np.cos(3*pi/8), C*np.cos(7*pi/16)
    A = np.array([\
                [a,a,a,a,a,a,a,a], \
                [b,d,e,g,-g,-e,-d,-b], \
                [c,f,-f,-c,-c,-f,f,c], \
                [d,-g,-b,-e,e,b,g,-d], \
                [a,-a,-a,a,a,-a,-a,a], \
                [e,-b,g,d,-d,-g,b,-e], \
                [f,-c,c,-f,-f,c,-c,f], \
                [g,-e,d,-b,b,-d,e,-g]])
    DCT_mat_on_vec = np.zeros((64,64))

    for i in range(8):
        DCT_mat_on_vec[i*8:i*8+8,i*8:i*8+8]=A

    idx =  np.arange(64)
    idx = idx.reshape((8,8),order='C')
    idx_T = idx.T
    idx_F = idx_T.flatten('C')
    mat_T = np.zeros((64,64))
    for i in range(64):
        mat_T[i,idx_F[i]]=1

    #DCT_T_vec = np.dot(mat_T,DCT_mat_on_vec)
    DCT_T_vec = np.dot(DCT_mat_on_vec,mat_T)


    T = np.dot(DCT_T_vec,DCT_T_vec)

    D_T = np.zeros((n_c_2**2,n_c_2**2))

    # M 
    
    for i in range(n_b**2):
        D_T[i*64: i*64+64, i*64: i*64+64] = T[:,:]
        
        
    Q = np.zeros_like(D_T)
    Q[np.diag_indices(D_T.shape[0])] = 1/np.array([c_quant]*(n_b**2)).flatten()
    return(Q @ D_T)


def generate_block_idx(block_n_list):
    idx = []
    for i in np.arange(block_n_list.size):
        idx = np.concatenate([idx, np.arange(block_n_list[i]*64,(block_n_list[i]+1)*64)]).astype(np.int)
    return(idx)
def generate_permuted_cov_small(C):

    idxLB1 = generate_block_idx(np.array([0,2,6,8,4]))
    idxLB2 = generate_block_idx(np.array([1,7,3,5,0,2,6,8,4]))
    idxLB3 = generate_block_idx(np.array([3,5,1,7,4]))
    idxLB4 = generate_block_idx(np.array([4]))
    permCLB1 = np.zeros((C.shape[0],C.shape[1], 5*64,5*64))
    permCLB2 = np.zeros((C.shape[0],C.shape[1], 9*64,9*64))
    permCLB3 = np.zeros((C.shape[0],C.shape[1], 5*64,5*64))
    permCLB4 = np.zeros((C.shape[0],C.shape[1], 1*64,1*64))
    
    #Lattice block Type 1
    for i in np.arange(1,C.shape[0],2):
        for j in np.arange(1,C.shape[1],2):
            permCLB1[i,j] = C[i,j, idxLB1,:][:, idxLB1]
    #Lattice block Type 2
    for i in np.arange(1,C.shape[0],2):
        for j in np.arange(0,C.shape[1],2):
            permCLB2[i,j] = C[i,j, idxLB2,:][:, idxLB2]
    #Lattice block Type 3
    for i in np.arange(0,C.shape[0],2):
        for j in np.arange(1,C.shape[1],2):
            permCLB3[i,j] = C[i,j, idxLB3,:][:, idxLB3]
    
    #Lattice block Type 4
    for i in np.arange(0,C.shape[0],2):
        for j in np.arange(0,C.shape[1],2):
            permCLB4[i,j] = C[i,j, idxLB4,:][:, idxLB4]
    
    
    permCLB1 = npl.cholesky(permCLB1[1::2,1::2])
    permCLB2 = npl.cholesky(permCLB2[1::2,::2])
    permCLB3 = npl.cholesky(permCLB3[::2,1::2])
    permCLB4 = npl.cholesky(permCLB4[::2,::2])
    chol_C = []
    k = [0,0,0,0]
    
    for i in np.arange(C.shape[0]):
        l = [0,0,0,0]
        chol_C.append([])
        for j in np.arange(C.shape[1]):
            if i % 2 == 1 and j % 2 == 1:
                #print(i,j,k[0],l[0])
                chol_C[i].append(permCLB1[k[0],l[0]])
                l[0]+=1
            elif i%2 == 1 and j % 2 == 0:
                chol_C[i].append(permCLB2[k[1],l[1]])
                l[1]+=1
            elif i%2 == 0 and j % 2 == 1:
                chol_C[i].append(permCLB3[k[2],l[2]])
                l[2]+=1
            else:
                chol_C[i].append(permCLB4[k[3],l[3]])
                #chol_C[i].append([])
                l[3]+=1
        if i % 2 == 1:
            k[0]+=1
            k[1]+=1
        else:
            k[2]+=1
            k[3]+=1

    return(np.array(chol_C, dtype=object))
def generate_graph_cholesky_structure_from_perm(chol_C):
    L1 = {}
    L2 = {}
    L3 = {}
    L4 = {}
    
    
    #Lattice block type 1
    for i in np.arange(1,chol_C.shape[0],6):
        for j in np.arange(1,chol_C.shape[1],6):
            ### L1 ###
            L1[i,j,0] = chol_C[i-1,j-1].astype(np.float32)
            L1[i,j,2] = chol_C[i-1,j+1].astype(np.float32)
            L1[i,j,6] = chol_C[i+1,j-1].astype(np.float32)
            L1[i,j,8] = chol_C[i+1,j+1].astype(np.float32)
            ### L2 ###
            L2[i,j,4] = chol_C[i,j][-64:, :].astype(np.float32)
            ### L3 ###
            L3[i,j,1] = chol_C[i-1,j][-64:, :].astype(np.float32)
            L3[i,j,7] = chol_C[i+1,j][-64:, :].astype(np.float32)
            
            ### L4 ###
            L4[i,j,5] = chol_C[i,j+1][-64:, :].astype(np.float32)
            L4[i,j,3] = chol_C[i,j-1][-64:, :].astype(np.float32)

    #Lattice block type 2
    for i in np.arange(1,chol_C.shape[0],6):

        for j in np.arange(4,chol_C.shape[1]-1,6):
            L1[i,j,1] = chol_C[i-1,j].astype(np.float32)
            L1[i,j,7] = chol_C[i+1,j].astype(np.float32)
            
            L2[i,j,3] = chol_C[i,j-1][-64:, :].astype(np.float32)
            L2[i,j,5] = chol_C[i,j+1][-64:, :].astype(np.float32)
            
            L3[i,j,0] = chol_C[i-1,j-1][-64:, :].astype(np.float32)
            L3[i,j,2] = chol_C[i-1,j+1][-64:, :].astype(np.float32)
            L3[i,j,6] = chol_C[i+1,j-1][-64:, :].astype(np.float32)
            L3[i,j,8] = chol_C[i+1,j+1][-64:, :].astype(np.float32)
            
            L4[i,j,4] = chol_C[i,j][-64:, :].astype(np.float32)
    #Lattice block type 3
    for i in np.arange(4,chol_C.shape[0]-1,6):

        for j in np.arange(1,chol_C.shape[1],6):
            L1[i,j,3] = chol_C[i,j-1].astype(np.float32)
            L1[i,j,5] = chol_C[i,j+1].astype(np.float32)
            
            L2[i,j,1] = chol_C[i-1,j][-64:, :].astype(np.float32)
            L2[i,j,7] = chol_C[i+1,j][-64:, :].astype(np.float32)
            
            L3[i,j,4] = chol_C[i,j][-64:, :].astype(np.float32)
            
            L4[i,j,0] = chol_C[i-1,j-1][-64:, :].astype(np.float32)
            L4[i,j,2] = chol_C[i-1,j+1][-64:, :].astype(np.float32)
            L4[i,j,6] = chol_C[i+1,j-1][-64:, :].astype(np.float32)
            L4[i,j,8] = chol_C[i+1,j+1][-64:, :].astype(np.float32)
            
    #Lattice block type 4
    for i in np.arange(4,chol_C.shape[0]-1,6):

        for j in np.arange(4,chol_C.shape[1]-1,6):
            L3[i,j,3] = chol_C[i,j-1][-64:, :].astype(np.float32)
            L3[i,j,5] = chol_C[i,j+1][-64:, :].astype(np.float32)
            
            L4[i,j,1] = chol_C[i-1,j][-64:, :].astype(np.float32)
            L4[i,j,7] = chol_C[i+1,j][-64:, :].astype(np.float32)
            
            L1[i,j,4] = chol_C[i,j].astype(np.float32)
            
            L2[i,j,0] = chol_C[i-1,j-1][-64:, :].astype(np.float32)
            L2[i,j,2] = chol_C[i-1,j+1][-64:, :].astype(np.float32)
            L2[i,j,6] = chol_C[i+1,j-1][-64:, :].astype(np.float32)
            L2[i,j,8] = chol_C[i+1,j+1][-64:, :].astype(np.float32)

    
    return(L1,L2,L3,L4)           
def estimate_cov(rb,a,b,spH,eps=10**-5,eps_max=1, lvl_max=2**16-1):
    spC_list = []
    for i in np.arange(rb.shape[0]):
        spC_list.append([])
        for j in np.arange(rb.shape[1]):
            v = a*rb[i,j] +b
            v[v < eps] = eps
            v[v > a*(95*lvl_max/100)+b] = eps_max
            spR= sp.diags(np.sqrt(v).astype(np.float32))
            spL = (spH @ spR)
            spC = spL@spL.T
            #spC = sparsify(spC.todense().astype(np.float32), alpha=0.99)
            spC_list[i].append(spC)
    return(spC_list)
def convert_dict_to_npy(L1f, L2f, L3f, L4f):
    L1 = np.zeros((len(L1_keys), 64, 64), dtype=np.float32)
    for i in range(len(L1_keys)):
        L1[i] = L1f[L1_keys[i]]


    L2 = np.zeros((len(L2_keys), 64, 320), dtype=np.float32)
    for i in range(len(L2_keys)):
        L2[i] = L2f[L2_keys[i]]


    L3 = np.zeros((len(L3_keys), 64, 320), dtype=np.float32)
    for i in range(len(L3_keys)):
        L3[i] = L3f[L3_keys[i]]


    L4 = np.zeros((len(L4_keys), 64, 576), dtype=np.float32)
    for i in range(len(L4_keys)):
        L4[i] = L4f[L4_keys[i]]
    return(L1,L2,L3,L4)
def quant_C(C, tquant):
    tmp = np.zeros((C.shape[0], C.shape[1], C[0,0].shape[0], C[0,0].shape[0]))
    for i in np.arange(C.shape[0]):
        for j in np.arange(C.shape[1]):
            tmp[i,j] = tquant@C[i,j].todense()@tquant.T
    return(tmp)

