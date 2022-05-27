# Time-stamp: <2017-06-15 10:56:32 zshuang>
# S=1 Heisenberg + (sz)^2 term

import sys
import os
if'../lib/' not in sys.path:
    sys.path.append('../lib/')
import numpy as np
import math
from scipy import sparse
from scipy import linalg as la
from scipy.sparse import linalg as sla
#from scipy.optimize import minimize as spmin
import scipy.optimize as so
#from npext import *
import npext
from helper import *
import cmath
import itertools as it
#from scipy.special import binom as binom
import functools
import collections

# Spin-1 matrices
SP = sparse.csr_matrix( np.sqrt(2) * np.diag(np.array([1,1],dtype=np.double), 1) )
#SM = sparse.csr_matrix( npext.dagger(SP) )
SM = SP.H
SZ = sparse.csr_matrix( np.diag(np.array([1,0,-1], dtype=np.double)) )
SX = (SP + SM)/2
ISY = (SP - SM)/2
# S kron S
SS = 0.5 * sparse.kron(SP,SM) + 0.5 * sparse.kron(SM,SP) + sparse.kron(SZ,SZ)
SS_sqr = SS.dot(SS)

def gen_sz_total_mask(nsites=10,sztot=0):
    """ generate index mask representing total Sz = sztot """
    # sgrid[s1,s2,...,sN] = total sz of conf [s1,s2,...sN], where si =
    # 0,1,2 for actual spin-z -1,0,1

    sgrid = np.meshgrid(*([np.arange(-1,2)]*nsites), sparse=True)
    sz = np.sum(sgrid, axis=0)
    return (sz == sztot).flatten()
def gen_sz0_indices(nsites=10):
    """ generate all integers representing total Sz singlets """
    return np.arange(3**nsites)[gen_sz_total_mask(nsites,0)]

def kron_reduce(*ops):
    """ take the kron product of all inputing operators, _in order_ """
    return functools.reduce(sparse.kron, ops)

def h_heis_proj(nsites):
    """ Heisenberg Hamiltonian. Projected onto total sz=0
    """
    #print('constructing Hamiltonian...')

    sz0 = gen_sz0_indices(nsites)
    dim = sz0.shape[0]
    res = sparse.csr_matrix((dim,dim),dtype=np.double)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        hi = kron_reduce(id1, SS, id2).tocsr()
        res += hi[sz0][:,sz0]

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    #sx = 0.5 * (SP + SM)
    #isy = 0.5 * (SP - SM)
    for op1,op2 in zip((SX,ISY,SZ),(SX,-ISY,SZ)):
        hi = kron_reduce(op1, id_mid, op2).tocsr()
        res += hi[sz0][:,sz0]
    #print('Hamlitonian constructed!')
    return res
def h_heis(nsites):
    """ Heisenberg Hamiltonian
    """
    #print('constructing Hamiltonian...')

    dim = 3**nsites
    res = sparse.csr_matrix((dim,dim),dtype=np.double)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        hi = kron_reduce(id1, SS, id2).tocsr()
        res += hi

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    #sx = 0.5 * (SP + SM)
    #isy = 0.5 * (SP - SM)
    for op1,op2 in zip((SX,ISY,SZ),(SX,-ISY,SZ)):
        hi = kron_reduce(op1, id_mid, op2).tocsr()
        res += hi
    #print('Hamlitonian constructed!')
    return res

def h_sz2_proj_slow(nsites=10):
    sz2 = SZ.dot(SZ)
    sz0 = gen_sz0_indices(nsites)
    dim = sz0.shape[0]
    res = sparse.csr_matrix((dim,dim), dtype=np.double)
    for i in np.arange(nsites):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 1 - i))
        hi = kron_reduce(id1, sz2, id2).tocsr()
        res += hi[sz0][:,sz0]
    return res

def h_sz2_proj(nsites=10):
    """ the sz^2 Hamiltonian, projected onto total sz=0 """

    sz2_1site = [1,0,1] # sz^2 value of a single site

    # sz2grid[sj][s1,s2,...,sN] = sj^2
    sz2grid = np.meshgrid(*([sz2_1site]*nsites), sparse=True)
    # sz2: the diagonal line of the sum(sz^2) operator
    sz2 = np.sum(sz2grid, axis=0).flatten()[gen_sz0_indices(nsites)]
    return sparse.diags(sz2, offsets=0, format='csr')

def h_sz2(nsites=10):
    """ the sz^2 Hamiltonian """

    sz2_1site = [1,0,1] # sz^2 value of a single site

    # sz2grid[sj][s1,s2,...,sN] = sj^2
    sz2grid = np.meshgrid(*([sz2_1site]*nsites), sparse=True)
    # sz2: the diagonal line of the sum(sz^2) operator
    sz2 = np.sum(sz2grid, axis=0).flatten()
    return sparse.diags(sz2, offsets=0, format='csr')

def svd_basis(*ubasis):
    """ carry out svd for a set of non-orthogonal basis states, each as a *row* vector """
    umat = np.asarray([ uu.flatten() for uu in ubasis ])
    return la.svd(umat, full_matrices = False)

def proj_ortho_basis(u, vmat):
    """ project u onto the space spanned by *rows* of vmat """
    u = u.flatten()
    coef = np.sum(u.conj() * vmat, axis=-1)

    weight = np.sum(coef * coef.conj())
    uproj = np.sum(coef.reshape(-1,1) * vmat, 0)
    uproj /= la.norm(uproj)
    return weight,uproj,coef
    
def proj_nonortho_basis(u, *ubasis):
    """ project the target state u to the space spanned by a set of non-orthogonal basis states ubasis """

    u = u.flatten()

    # Ortho-normalize ubasis. Rows of vv form ortho-normal basis of the
    # umat space
    #umat = np.asarray([ uu.flatten() for uu in ubasis ])
    #uu,dd,vv = la.svd(umat, full_matrices=False)
    uu,dd,vv = svd_basis(ubasis)
    return proj_ortho_basis(u, vv)

def full_wf(uproj, nsites):
    up = uproj.flatten()
    res = np.zeros(3**nsites)
    res[gen_sz0_indices(nsites)] = up
    return res
        
def ent_decomp(uproj, nsites, ncut):
    """ given a total sz=0 projected wavefunction, compute the left-right entanglement decomposition at ncut """
    ufull = full_wf(uproj, nsites)
    return la.svd(ufull.reshape(3**ncut, -1), full_matrices=False)

class Hjd():
    def __init__(self, nsites = 12, proj=True):
        """ if proj == True, then project onto total sz=0 """
        sz0idx = gen_sz0_indices(nsites)
        self.sz0idx = sz0idx
        self.nsites = nsites
        if proj == True:
            self.proj = True
            self.h_heis = h_heis_proj(nsites)
            self.h_sz2 = h_sz2_proj(nsites)
        else:
            self.proj = False
            self.h_heis = h_heis(nsites)
            self.h_sz2 = h_sz2(nsites)

    def h(self, d):
        return self.h_heis + d*self.h_sz2

    def h_with_eig(self,d=0):
        """ return Hamiltonian and ground state and energy at beta """
        hh = self.h(d)

        eig,u = sla.eigsh(hh, k=1, which='SA')

        return eig[0],u.flatten(),hh

    def export_erg2(self, fn='erg', dfrom = -2, dto = 2, dd = 0.1):
        """ compute the lowest two energies """

        with open_for_write(fn + '.dat') as dat:
            for d in np.arange(dfrom, dto, dd):
                print('\rd = %g                '%d, end='')
                hh = self.h(d)
                eig,u = sla.eigsh(hh, k=2, which='SA')
                print('%g\t%g\t%g'%(d, *eig), file=dat)
    def export_wf(self, fn='wf_all_D', dfrom = 0, dto = 2, dd = 0.01):
        """ save the wavefunctions in a range of beta for later use """

        txt = """
nsites = %d
dfrom = %g
dto = %g
dd = %g """%(self.nsites, dfrom, dto, dd)
        write_to_file(fn + '.par', txt)

        all_gs = []
        for d in np.arange(dfrom, dto, dd):
            print('\rd = %g                '%d, end='')
            eig,u,h = self.h_with_eig(d)
            all_gs.append(u.flatten())

        np.save(fn + '.npy', np.asarray(all_gs))
        np.savetxt(fn + '.ds', np.arange(dfrom, dto, dd))

    def load_wf(self, fn):
        """ load wavefunctions saved via export_wf """
        self.d_all = np.loadtxt(fn + '.ds')
        self.wf_all = np.load(fn + '.npy')
        
    def export_phase_space_svd_sparse_d(self, fn = 'phase_space_svd_sparse_d', bases_d=None):
        """ assume all wavefunctions are saved in self.wf_all, as *row* vectors, carry out svd using basis states at d = bases_d. Then use the orthonormalized bases to expand all wavefunctions """

        if bases_d == None:
            bases_d = self.d_all
        bases_idx = [ self.d_all.searchsorted(d) for d in bases_d ]
        actual_ds = self.d_all[bases_idx]
        uu,dd,vv = la.svd(self.wf_all[bases_idx], full_matrices = False)

        header = gen_header('d sval')
        with open_for_write(fn + '_sval.dat') as dat_sval:
            print('# two columns not related, i.e., sval not a fn of d', file=dat_sval)
            print(header, file=dat_sval)
            for b,d in zip(actual_ds, dd):
                print('%g\t%g'%(b,d), file=dat_sval)

        header = gen_header('d total_w coef1 coef2 ...')
        with open_for_write(fn + '.dat') as dat:
            print(header, file=dat)
            for d, u in zip(self.d_all, self.wf_all):
                print('\rd = %g                '%d, end='')
                w,uproj,coef = proj_ortho_basis(u, vv)
                print('%g\t%g\t%s'%(d, w, '\t'.join(['%g'%d for d in coef])), file=dat)

    def export_proj(self, bases, fn='var_proj'):
        """ assume all wavefuncntions are saved in self.wf_all, as *row* vectors, project them onto a given orthonormal bases """
        header = gen_header('d total_w coef1 coef2 ...')
        with open_for_write(fn + '.dat') as dat:
            print(header, file=dat)
            for d, u in zip(self.d_all, self.wf_all):
                print('\rd = %g                '%d, end='')
                w,uproj,coef = proj_ortho_basis(u, bases)
                print('%g\t%g\t%s'%(d, w, '\t'.join(['%g'%d for d in coef])), file=dat)
                
