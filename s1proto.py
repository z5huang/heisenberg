# S=1 Heisenberg
# prototyping

import sys
import os
if'../lib/' not in sys.path:
    sys.path.append('../lib/')
import numpy as np
import math
from scipy import sparse
#from numpy import linalg as la
from scipy import linalg as la
#from scipy.optimize import minimize as spmin
import scipy.optimize as so
#from npext import *
import npext
import cmath
import itertools as it
#from scipy.special import binom as binom

def int2spin(n, nsites=0):
    """ convert an integer to a spin 1 configuration.

    if nsite > 0, pad -1 to the left for an nsite configuration
    """
    repr = np.array(list(np.base_repr(n, 3)), dtype=int)
    if nsites <= 0:
        return repr - 1
    else:
        l = len(repr)
        if l > nsites: # more spins than sites
            print('l = %d, nsites = %d, there are more spins than sites. Returning the rightmost %d spins'%(l,nsites,nsites))
            return repr - 1
        res = np.zeros(nsites,dtype=int)
        res[-l:] = repr
        return res - 1
def spin2int(s):
    """ convert a spin 1 configuration to an integer """
    base3 = s+1 # from {-1,0,1} to {0,1,2}
    srep =  np.array2string(base3,separator='')[1:-1]
    return int(srep, 3)

def s_dot_s(i,j,s):
    """ compute s_i dot s_j acting on a spin configuration (actual array of [-1,0,1]s).

    returns (coefs, states), where states is a list of resulting configurations, and coefs are the corresponding coefficients.
    """

    si,sj = s[i],s[j]
    coefs = []
    states = []

    ## NB:
    # S^+ = sqrt(2) x (0 1 0)
    #                 (0 0 1)
    #                 (0 0 0) 

    # compute 1/2 (si^+ sj^-)
    if (si != 1) and (sj != -1):
        s_res = s.copy()
        s_res[i] += 1
        s_res[j] -= 1
        coefs.append(np.sqrt(0.5))
        states.append(spin2int(s_res))
    # compute 1/2 (si^- sj^+)
    if (si != -1) and (sj != 1):
        s_res = s.copy()
        s_res[i] -= 1
        s_res[j] += 1
        coefs.append(np.sqrt(0.5))
        states.append(spin2int(s_res))
    # compute si^z sj^z
    if si * sj != 0:
        coefs.append(si*sj)
        states.append(spin2int(s))

    return (coefs,states)

def h(nsites=12):
    """ construct the Hamiltonian matrix 


    WRONG FOR NOW. some elements differ by sqrt(2). Compare with h_kron below
    """

    dim = 3**nsites

    cols = []
    rows = []
    elts = []
    for col in range(dim):
        s = int2spin(col, nsites)
        for i in range(nsites-1):
            coefs, states = s_dot_s(i,i+1,s)
            cols.append( [col]*len(coefs) )
            rows.append( states )
            elts.append( coefs )
        # boundary link
        coefs, states = s_dot_s(nsites-1,0,s)
        cols.append( [col]*len(coefs) )
        rows.append( states )
        elts.append( coefs )
        

    # flatten the arrays
    cols = list(it.chain(*cols))
    rows = list(it.chain(*rows))
    elts = list(it.chain(*elts))

    ## coo matrix allows for duplicate entries (and takes the
    ## sum). E.g., if element (1,1) is specified 3 times, with values
    ## v1,v2,v3, then the eventual element is the sum, v1+v2+v3
    res = sparse.coo_matrix( (elts, (rows,cols)), shape=(dim,dim) )
    return res

def s_kron_s():
    """ nearest neighbor spin dot """
    sp = sparse.csr_matrix( np.sqrt(2) * np.diag(np.array([1,1],dtype=np.double), 1) )
    sm = sparse.csr_matrix( npext.dagger(sp) )
    sz = sparse.csr_matrix( np.diag(np.array([1,0,-1], dtype=np.double)) )

    return (0.5 * sparse.kron(sp,sm) + 0.5 * sparse.kron(sm,sp) + sparse.kron(sz,sz), sp,sm,sz)
def h_kron(nsites):
    print('constructing Hamiltonian...')
    res = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    ss,sp,sm,sz = s_kron_s()
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        res += sparse.kron(id1, sparse.kron(ss, id2))

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    sx = 0.5 * (sp + sm)
    isy = 0.5 * (sp - sm)
    for op1,op2 in zip((sx,isy,sz),(sx,-isy,sz)):
        res += sparse.kron(op1, sparse.kron(id_mid, op2))
    print('Hamlitonian constructed!')
    return res

def h_aklt(nsites):
    res = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    ss,sp,sm,sz = s_kron_s()
    pp = ss + ss.dot(ss)/3 + 2/3*sparse.identity(3**2)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        res += sparse.kron(id1, sparse.kron(pp, id2))

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    sx = 0.5 * (sp + sm)
    isy = 0.5 * (sp - sm)
    s_s = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    for op1,op2 in zip((sx,isy,sz),(sx,-isy,sz)):
        s_s += sparse.kron(op1, sparse.kron(id_mid, op2))
    p_p = s_s + s_s.dot(s_s)/3 + 2/3*sparse.identity(3**nsites)
    res += p_p
    return res

#### NB ####
## To obtain the eigenvalue, use sparse.linalg.eigsh(h, 1, which='SA')
## Here, 'SA' is important, it finds "the smallest eigenvalue inclusive of any negative sign"
