# Time-stamp: <2018-01-13 17:43:39 zshuang>
# S=1 bilinear biquadratic, Entanglement Geometry

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

import pandas as pd
import json

from sklearn.manifold import Isomap
from sklearn.utils import check_array
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path

def mydist(x,y):
    return np.abs(np.dot(x,y))
class MyIsomap(Isomap):
    """ modify neighborhood calling to allow user supplied metric """
    def __init__(self,metric='correlation', **param):
        super().__init__(**param)
        self.metric =  metric
    def _fit_transform(self, X):
        """ copied from github source code version Nov. 25 2016, with metric modification """
        X = check_array(X)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      n_jobs=self.n_jobs,
                                      metric = self.metric   # enable custom metric
                                          )
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)

        kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                               mode='distance', n_jobs=self.n_jobs,
                               metric = self.metric # enable custom metric
                                   )

        self.dist_matrix_ = graph_shortest_path(kng,
                                                method=self.path_method,
                                                directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)

## sample fitting code
def fit_ent(uproj, nsites, ncut, metric=mydist, cutoff=1e-10):
    ufull = full_wf(uproj, nsites)
    mat = ufull.reshape(3**ncut, -1)
    norm2 = np.sum(mat * mat.conj(), axis=0) # norm squared of each column
    mat_red = mat[:, norm2 > cutoff] # throw away states with extremely small norms, possiblely due to them being actual zeros
    msg('reduced number of columns from ', mat.shape[1], ' to ', mat_red.shape[1])
    iso = MyIsomap(n_neighbors = 6, n_components = 2, n_jobs = -1, metric = metric)
    iso.fit(mat_red.T) # iso treat rows as input vectors
    return iso.transform(mat_red.T)
    

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


def s1ex(wf,i,j,nsites=None):
    """ given a wavefunction wf, exchange the ith and the jth sites """
    shape = wf.shape
    if isNone(nsites):
        nsites = int(round( math.log(np.prod(shape)) / math.log(3)))
    return np.swapaxes(wf.reshape([3]*nsites), i,j).reshape(shape)

def vortex(wf,start=0,lev=1,nsites=None):
    """ apply a vortex to a wf """
    shape = wf.shape
    if isNone(nsites):
        nsites = int(round( math.log(np.prod(shape)) / math.log(3) ))

    vsize = 2*lev
    res = wf.reshape([3] * nsites)

    for i in range(start, start + lev, 2):
        j = vsize - 1 - i
        res = np.swapaxes(res, i,j)
    return res

def homogenize(wf, nsites=None):
    """ make a wavefunction translationally invariant """
    shape = wf.shape
    if isNone(nsites):
        nsites = int(round( math.log(np.prod(shape)) / math.log(3) ))

    ww = wf.reshape([3]*nsites)
    #msg(ww)
    res = 0
    for _ in it.repeat(None, nsites):
        ww = np.rollaxis(ww, -1)
        res += ww
        #msg(la.norm(res.flatten()))
    res = res.flatten()
    return (res / la.norm(res)).reshape(shape)
    
def gen_sz_total_mask(nsites=10,sztot=0):
    """ generate index mask representing total Sz = sztot """
    # sz[s1,s2,...,sN] = total sz of conf [s1,s2,...sN], where si =
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

def h_aklt_proj(nsites, bc_sign=1):
    """ AKLT Hamiltonian. Projected onto total sz=0
    
    bc_sign: sign of boundary condition
    obc: if use open boundary
    """

    sz0 = gen_sz0_indices(nsites)
    dim = sz0.shape[0]
    res = sparse.csr_matrix((dim,dim),dtype=np.double)

    pp = SS + SS_sqr/3 + 2/3*sparse.identity(3**2)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        hi = kron_reduce(id1, pp, id2).tocsr()
        res += hi[sz0][:,sz0]

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    #sx = 0.5 * (SP + SM)
    #isy = 0.5 * (SP - SM)
    s_s = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    for op1,op2 in zip((SX,ISY,SZ),(SX,-ISY,SZ)):
        s_s += kron_reduce(op1, id_mid, op2)
    p_p = (  s_s + s_s.dot(s_s)/3 + 2/3*sparse.identity(3**nsites)  ).tocsr()
    res += bc_sign * p_p[sz0][:,sz0]
    return res

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
    res = np.zeros(3**nsites, dtype=uproj.dtype)
    res[gen_sz0_indices(nsites)] = up
    return res
        
def ent_decomp(uproj, nsites, ncut):
    """ given a total sz=0 projected wavefunction, compute the left-right entanglement decomposition at ncut (# of left chain) """
    ufull = full_wf(uproj, nsites)
    return la.svd(ufull.reshape(3**ncut, -1), full_matrices=False)


def quantum_distance(neighbor_overlap):
    """ given an ordered list of overlap of neighboring states, compute the quantum distance """
    d_distance = np.sqrt(1 - np.abs(neighbor_overlap)**2) # distance between psi1 and psi2 is sqrt(1 - <psi1 | psi2><psi2 | psi1>)
    return np.sum(d_distance), d_distance
    

def quantum_distance_from_states(states):
    """ given a path of states, compute the quantum distance of this path 

    states: states[i] is the ith state
    """

    neighbor_overlap = np.sum(states[:-1] * states[1:].conj(), axis=1)[:-1]
    return quantum_distance(neighbor_overlap)


def quantum_distance_from_overlap_mat(overlap_mat):
    """ given the overlap matrix of a path of states, compute the quantum distance

    overlap_mat[i,j] is the overlap between states i and j
    """
    neighbor_overlap = np.diag(overlap_mat, 1) # 1st superdiagonal
    return quantum_distance(neighbor_overlap)
    

def export_dvh(uproj, nsites, ncut, fn='s1-ent-dvh'):
    """ Compute decomposition of truncated states in the entanglement eigenbases """

    ufull = full_wf(uproj, nsites)
    mm = ufull.reshape(3**ncut, -1)
    # each column of mm represents a left-chain state labelled by a right-chain configuration
    u, d, vh = la.svd(mm, full_matrices=False)

    dvh_T = vh.T * d
    # dvh_T = (uh . mm)^T. #
    # Each column of dvh is a left-chain state written
    # in the singular vector basis (columns of u). Note that these
    # left-chain states are *not* normalized.
    np.savetxt(fn + '.dat', dvh_T, delimiter='\t', header='k=1\tk=2\tk=3\t...')
    

class Hvar():
    def __init__(self, nsites = 12, u_aklt = None):

        if isNone(u_aklt):
            u_aklt = sla.eigsh(h_aklt_proj(nsites), k=1, which='SA')[1].flatten()
        #if u_target == None:
        #    u_target = sla.eigsh(h_heis_proj(nsites), k=1, which='SA')[1].flatten()

        sz0idx = gen_sz0_indices(nsites)
        u_aklt_full = np.zeros(3**nsites, dtype=np.double)
        u_aklt_full[sz0idx] = u_aklt

        self.sz0idx = sz0idx
        self.nsites = nsites
        self.u_aklt_full = u_aklt_full
        self.u_aklt = u_aklt
        #self.u_target = u_target

    def u_ex(self, *sites):
        """ Apply exchanges to the aklt wavefunction, and homogenize (translation symmetrization). Returns a wf projected to total sz=0

        sites: = s1, s2, s3, ..., where si = 0, 1, 2. Exchange will be carried out on [s1,s1+1], [s2,s2+1], ..., in that order
        """
        res = self.u_aklt_full.copy()
        for s in sites:
            res = s1ex(res, s, (s+1)%self.nsites, self.nsites)
            #msg('s = ',s,' , s+1 = ', (s+1)%self.nsites)
        res = homogenize(res, self.nsites)
        return res[self.sz0idx]
    def u_ex_nohomo_full(self, *sites):
        """ Apply exchanges to the aklt wavefunction, and homogenize (translation symmetrization). Returns a wf projected to total sz=0

        sites: = s1, s2, s3, ..., where si = 0, 1, 2. Exchange will be carried out on [s1,s1+1], [s2,s2+1], ..., in that order
        """
        res = self.u_aklt_full.copy()
        for s in sites:
            res = s1ex(res, s, (s+1)%self.nsites, self.nsites)
            #msg('s = ',s,' , s+1 = ', (s+1)%self.nsites)
        return res
        
class Hinter_var(Hvar):
    """ interpolating with the AKLT beta """
    def __init__(self, nsites = 12, u_aklt = None):
        super().__init__(nsites, u_aklt)
        self.h1,self.h2 = self.memo_h_terms()

    def memo_h_terms(self):
        """ memoize bilinear and bi-quadratic terms, projected to sz=0 """

        sz0 = self.sz0idx

        h1 = 0
        h2 = 0

        # bulk
        for i in range(self.nsites - 1):
            id1 = sparse.identity(3**i, dtype=np.double)
            id2 = sparse.identity(3**(self.nsites - 2 - i), dtype=np.double)
            h1i = kron_reduce(id1, SS, id2).tocsr()[sz0][:,sz0]
            h1 += h1i
            h2 += h1i.dot(h1i)
            
        # boundary link
        id_mid = sparse.identity(3**(self.nsites - 2))
        s_s = (0.5 * kron_reduce(SP,id_mid,SM) + 
               0.5 * kron_reduce(SM,id_mid,SP) + 
                     kron_reduce(SZ,id_mid,SZ)).tocsr()[sz0][:,sz0]
        h1 += s_s
        h2 += s_s.dot(s_s)

        return h1,h2

    def gen_ortho_basis(self):
        """ generate orthogonal basis with 1 and 2 exchanges """
        u1 = self.u_ex(0)
        u2 = [ self.u_ex(0,i) for i in range(1, self.nsites//2 + 1) ]
    
        v1 = svd_basis(self.u_aklt, u1)[-1]
        v2 = svd_basis(self.u_aklt, u1, *u2)[-1]
    
        return v1,v2
    def u_ex_multi_basis_old(self, n=1, sv_cutoff = 1e-6):
        """ return orthonormal basis of states with n exchanges """
        all_u = [ self.u_ex(*s) for s in it.combinations(range(self.nsites),n) ]
        u,d,v = svd_basis(*all_u)
        return v[(d**2/np.sum(d**2)) > sv_cutoff]
    
    def u_ex_multi_basis(self, n=1, sv_cutoff = 1e-6, return_sv = False):
        """return orthonormal basis of states with n exchanges

        Note that when two exchanges are next to each other, their order
        will matter, but we only included a state obtained by their input the order, thus we are probably missing some basis states there
        """

        # fix the first switch to occur at 0. There is still a huge degeneracy due to translation
        all_u = [ self.u_ex(0, *s) for s in it.combinations(range(1,self.nsites),n-1) ]
        u,d,v = svd_basis(*all_u)

        mask = (d**2/np.sum(d**2)) > sv_cutoff
        if return_sv:
            return v[mask], (d**2)[mask]
        else:
            return v[mask]

    def u_ex_multi_uniform(self, n=1, sv_cutoff = 1e-6):
        """ a single state consisting of uniform superposition of n exchanges

        Note that when two exchanges are next to each other, their order
        will matter, but we only included a state obtained by their input the order, thus we are probably missing some basis states there
        """

        res = 0
        for s in it.combinations(range(self.nsites), n):
            res += self.u_ex_nohomo_full(*s)
        return homogenize(res, self.nsites)[self.sz0idx]

    def u_ex_multi_consecutive_basis(self, n=1, sv_cutoff = 1e-6):
        """return orthonormal basis of states with n *consecutive* exchanges

        Note that when two exchanges are next to each other, their order
        will matter, but we only included a state obtained by their input the order, thus we are probably missing some basis states there
        """
        all_u = [ self.u_ex(*s) for s in it.permutations(range(n)) ]
        u,d,v = svd_basis(*all_u)
        return v[(d**2/np.sum(d**2)) > sv_cutoff]

    def h_with_eig(self,beta=0):
        """ return Hamiltonian and ground state and energy at beta """
        hh = self.h1 + beta * self.h2

        eig,u = sla.eigsh(hh, k=1, which='SA')

        return eig[0],u.flatten(),hh

    def gen_ex_bases(self, nex_max = 3, sv_cutoff = 1e-6):
        """ generate the orthonormal basis states with up to nex_max exchanges """
        msg('Computing basis for each exchange sector ...')
        bases_ex = [ self.u_ex_multi_basis(n, sv_cutoff) for n in range(1, nex_max+1) ]
        dims_ex = [ len(b) for b in bases_ex ]

        msg('Merging bases ...')
        bases = [self.u_aklt]
        bases_used = [self.u_aklt]
        dims_tot = []
        for n in range(nex_max):
            bases_used = [*bases_used, *bases_ex[n]]
            uu,dd,vv = svd_basis(*bases_used)

            vv = vv[ (dd**2/np.sum(dd**2)) > sv_cutoff ]
            bases.append( vv )
            dims_tot.append(vv.shape[0])

        return bases, dims_tot, bases_ex, dims_ex
        
    def export_proj(self, fn = 'var_proj', nex_max = 3, betafrom = -0.99, betato = 1, dbeta = 0.01, sv_cutoff = 1e-6):
        """ export projection onto variational Hilbert space spanned by "vortex" states up to nex_max nearest neighbor exchanges

        betafrom, betato, dbeta: list of beta to compute
        sv_cutoff: lower bound cutoff for singular values
        """


        txt = """
nsites = %d
betafrom = %g
betato = %g
dbeta = %g
nex_max = %d
"""%(self.nsites, betafrom, betato, dbeta,nex_max)
        write_to_file(fn + '.par', txt)

        header_f = gen_header('beta f0 f1 f2 f3 ...')
        header_e = gen_header('beta erg_exact erg_0 erg_1 erg_2 erg_3 ...')

        dat_f = open(fn + '_fidelity.dat', gen_mode(fn + '_fidelity.dat'))
        dat_e = open(fn + '_erg.dat', gen_mode(fn + '_erg.dat'))

        print(header_f, file=dat_f)
        print(header_e, file=dat_e)

        # bases_ex[i] = ortho-normal bases with i+1 nearest neighbor exchanges
        msg('Computing basis for each exchange sector ...')
        bases_ex = [ self.u_ex_multi_basis(n, sv_cutoff) for n in range(1, nex_max+1) ]
        dims_ex = [ len(b) for b in bases_ex ]
        write_to_file(fn + '_dims_ex.dat', '\n'.join(['%d'%d for d in dims_ex]))

        # bases[i] = ortho-normal basis with up to i nearest neighbor exchanges
        msg('Merging bases ...')
        bases = [self.u_aklt]
        bases_used = [self.u_aklt]
        dims_tot = []
        for n in range(nex_max):
            bases_used = [*bases_used, *bases_ex[n]]
            uu,dd,vv = svd_basis(*bases_used)

            vv = vv[ (dd**2/np.sum(dd**2)) > sv_cutoff ]
            bases.append( vv )
            dims_tot.append(vv.shape[0])
        write_to_file(fn + '_dims_tot.dat', '\n'.join(['%d'%d for d in dims_tot]))
        

        for beta in np.arange(betafrom, betato, dbeta):
            print('\rbeta = %g                '%beta, end='')
            eig,u,h = self.h_with_eig(beta)

            print('%g\t'%beta, end='', file=dat_f)
            print('%g\t'%beta, end='', file=dat_e)

            all_w = []
            all_erg = [eig]

            for bb in bases:
                wn, uprojn = proj_ortho_basis(u, bb)[:2]
                ergn = np.dot(uprojn, h.dot(uprojn))

                all_w.append(wn)
                all_erg.append(ergn)

            print('\t'.join(['%g'%val for val in np.sqrt(all_w)]), file=dat_f)
            print('\t'.join(['%g'%val for val in all_erg]), file=dat_e)            

        dat_f.close()
        dat_e.close()

#        v1, v2 = self.gen_ortho_basis() # basis with aklt + 1 and 2 exchanges
#        
#        with open(fn + '.dat', gen_mode(fn + '.dat')) as dat:
#            print(header, file=dat)
#            for beta in np.arange(betafrom, betato, dbeta):
#                print('\rbeta = %g                '%beta, end='')
#                eig,u,h = self.h_with_eig(beta)
#
#                w0, uproj0 = proj_ortho_basis(u, self.u_aklt)[:2]
#                w1, uproj1 = proj_ortho_basis(u, v1)[:2]
#                w2, uproj2 = proj_ortho_basis(u, v2)[:2]
#
#                erg0 = np.dot(uproj0, h.dot(uproj0))
#                erg1 = np.dot(uproj1, h.dot(uproj1))
#                erg2 = np.dot(uproj2, h.dot(uproj2))
#
#                print(('%g\t'*8)%(beta, w0,w1,w2,eig,erg0,erg1,erg2), file=dat)

    def export_proj_ent(self, fn = 'proj_ent', nex_max = 4, sv_cutoff = 1e-6, es_cutoff=60):
        """ export the entanglement spectrum of projected states onto the variational Hilbert space spanned by "vortex" states up to nex_max nearest neighbor exchanges

        assume self.load_wf has been invoked

        sv_cutoff: lower bound cutoff for singular values
        es_cutoff: max number of entanglement levels to export. If < 0, then write all
        """


        txt = """
nsites = %d
nex_max = %d
es_cutoff = %d
"""%(self.nsites, nex_max, es_cutoff)
        write_to_file(fn + '.par', txt)

        header = gen_header('beta n exact f0 f1 f2 f3 ...')
        dat = open(fn + '.dat', gen_mode(fn + '.dat'))
        print(header, file=dat)

        # bases_ex[i] = ortho-normal bases with i+1 nearest neighbor exchanges
        msg('Computing basis for each exchange sector ...')
        bases_ex = [ self.u_ex_multi_basis(n, sv_cutoff) for n in range(1, nex_max+1) ]
        dims_ex = [ len(b) for b in bases_ex ]
        write_to_file(fn + '_dims_ex.dat', '\n'.join(['%d'%d for d in dims_ex]))

        # bases[i] = ortho-normal basis with up to i nearest neighbor exchanges
        msg('Merging bases ...')
        bases = [self.u_aklt]
        bases_used = [self.u_aklt]
        dims_tot = []
        for n in range(nex_max):
            bases_used = [*bases_used, *bases_ex[n]]
            uu,dd,vv = svd_basis(*bases_used)

            vv = vv[ (dd**2/np.sum(dd**2)) > sv_cutoff ]
            bases.append( vv )
            dims_tot.append(vv.shape[0])
        write_to_file(fn + '_dims_tot.dat', '\n'.join(['%d'%d for d in dims_tot]))
        

        for beta,u in zip(self.beta_all, self.wf_all):
            print('\rbeta = %g                '%beta, end='')

            es_exact = ent_decomp(u, self.nsites, self.nsites//2)[1]
            all_es = [ es_exact ]

            for bb in bases:
                wn, uprojn = proj_ortho_basis(u, bb)[:2]
                es_uproj = ent_decomp(uprojn, self.nsites, self.nsites//2)[1]

                all_es.append(es_uproj)

            all_es = np.asarray(all_es)[:, :es_cutoff]
            for es_idx in np.arange(all_es.shape[1]):
                print('%g\t%g\t'%(beta,es_idx), file=dat, end='')
                print('\t'.join(['%g'%val for val in all_es[:, es_idx]]), file=dat)
        dat.close()

    def export_proj_2(self, fn = 'var_proj', nex_max = 3, betafrom = -0.99, betato = 1, dbeta = 0.01, sv_cutoff = 1e-6):
        """ export projection onto the variational Hilbert space spanned by states with up to nex_max exchanges

        Similar to self.export_proj, but use different weighting schemes for the vortex states, as controlled by the function used in generating bases_ex """ 

        txt = """
nsites = %d
betafrom = %g
betato = %g
dbeta = %g
nex_max = %d
"""%(self.nsites, betafrom, betato, dbeta, nex_max)
        write_to_file(fn + '.par', txt)

        header_f = gen_header('beta f0 f1 f2 f3 ...')
        header_e = gen_header('beta erg_exact erg_0 erg_1 erg_2 erg_3 ...')

        dat_f = open(fn + '_fidelity.dat', gen_mode(fn + '_fidelity.dat'))
        dat_e = open(fn + '_erg.dat', gen_mode(fn + '_erg.dat'))

        print(header_f, file=dat_f)
        print(header_e, file=dat_e)

        # bases_ex[i] = ortho-normal bases with i+1 nearest neighbor exchanges
        msg('Computing basis for each exchange sector ...')


        bases_ex = [ self.u_ex_multi_consecutive_basis(n, sv_cutoff) for n in range(1, nex_max+1) ]
        #bases_ex = [ [self.u_ex(*(2*np.arange(n)))] for n in range(1,nex_max + 1)] # sep2
        #bases_ex = [ [self.u_ex_multi_uniform(n)] for n in range(1, nex_max+1)]



        dims_ex = [ len(b) for b in bases_ex ]
        write_to_file(fn + '_dims_ex.dat', '\n'.join(['%d'%d for d in dims_ex]))

        # bases[i] = ortho-normal basis with up to i nearest neighbor exchanges
        msg('Merging bases ...')
        bases = [self.u_aklt]
        bases_used = [self.u_aklt]
        dims_tot = []
        for n in range(nex_max):
            bases_used = [*bases_used, *bases_ex[n]]
            uu,dd,vv = svd_basis(*bases_used)

            vv = vv[ (dd**2/np.sum(dd**2)) > sv_cutoff ]
            bases.append( vv )
            dims_tot.append(vv.shape[0])
        write_to_file(fn + '_dims_tot.dat', '\n'.join(['%d'%d for d in dims_tot]))
        

        for beta in np.arange(betafrom, betato, dbeta):
            print('\rbeta = %g                '%beta, end='')
            eig,u,h = self.h_with_eig(beta)

            print('%g\t'%beta, end='', file=dat_f)
            print('%g\t'%beta, end='', file=dat_e)

            all_w = []
            all_erg = [eig]

            for bb in bases:
                wn, uprojn = proj_ortho_basis(u, bb)[:2]
                ergn = np.dot(uprojn, h.dot(uprojn))

                all_w.append(wn)
                all_erg.append(ergn)

            print('\t'.join(['%g'%val for val in np.sqrt(all_w)]), file=dat_f)
            print('\t'.join(['%g'%val for val in all_erg]), file=dat_e)            

        dat_f.close()
        dat_e.close()
    def export_erg2(self, fn='erg', betafrom = -0.99, betato = 1, dbeta = 0.01):
        """ compute the lowest two energies """

        with open_for_write(fn + '.dat') as dat:
            for beta in np.arange(betafrom, betato, dbeta):
                print('\rbeta = %g                '%beta, end='')
                hh = self.h1 + beta * self.h2
                eig,u = sla.eigsh(hh, k=2, which='SA')
                print('%g\t%g\t%g'%(beta, *eig), file=dat)
    def export_phase_space_svd(self, fn='phase_space_svd', betafrom = -1, betato = 0.8, dbeta = 0.01):
        """ compute the singular values from the ground states in a beta range

        """


        txt = """
nsites = %d
betafrom = %g
betato = %g
dbeta = %g """%(self.nsites, betafrom, betato, dbeta)
        write_to_file(fn + '.par', txt)

        all_gs = []
        for beta in np.arange(betafrom, betato, dbeta):
            print('\rbeta = %g                '%beta, end='')
            eig,u,h = self.h_with_eig(beta)
            all_gs.append(u.flatten())
        uu,dd,vv = la.svd(np.asarray(all_gs), full_matrices=False)

        ud = uu*dd

        np.savetxt(fn + '_d.dat', dd)
        np.savetxt(fn + '_ud.dat', ud)

        return all_gs
    def export_phase_space_svd_sparse_beta_old(self, fn='phase_space_svd_sparse_beta', svd_beta = [-0.75,-0.25,0.25,0.75], betafrom = -1, betato = 0.8, dbeta = 0.01):
        """ compute the singular values from the ground states with svd_beta, but test projection for a denser beta settings

        """


        txt = """
nsites = %d
betafrom = %g
betato = %g
dbeta = %g 
svd_beta = '%s'
"""%(self.nsites, betafrom, betato, dbeta, ','.join([ '%g'%d for d in svd_beta]))
        write_to_file(fn + '.par', txt)

        all_gs = []
        for beta in svd_beta:
            print('\rbeta = %g                '%beta, end='')
            eig,u,h = self.h_with_eig(beta)
            all_gs.append(u.flatten())
        uu,dd,vv = la.svd(np.asarray(all_gs), full_matrices=False)

        header = gen_header('beta sval')
        with open_for_write(fn + '_sval.dat') as dat_sval:
            for b,d in zip(svd_beta, dd):
                print('%g\t%g'%(b,d), file=dat_sval)

        header = gen_header('beta total_w coef1 coef2 ...')
        with open_for_write(fn + '.dat') as dat:
            print(header, file=dat)
            for beta in np.arange(betafrom, betato, dbeta):
                print('\rbeta = %g                '%beta, end='')
                eig,u,h = self.h_with_eig(beta)
                w,uproj,coef = proj_ortho_basis(u, vv)
                print('%g\t%g\t%s'%(beta, w, '\t'.join(['%g'%d for d in coef])), file=dat)

    def export_wf(self, fn='wf_all_beta', betafrom = -1, betato = 0.8, dbeta = 0.01):
        """ save the wavefunctions in a range of beta for later use """

        txt = """
nsites = %d
betafrom = %g
betato = %g
dbeta = %g """%(self.nsites, betafrom, betato, dbeta)
        write_to_file(fn + '.par', txt)

        all_gs = []
        for beta in np.arange(betafrom, betato, dbeta):
            print('\rbeta = %g                '%beta, end='')
            eig,u,h = self.h_with_eig(beta)
            all_gs.append(u.flatten())

        np.save(fn + '.npy', np.asarray(all_gs))
        np.savetxt(fn + '.betas', np.arange(betafrom, betato, dbeta))

    def load_wf(self, fn):
        """ load wavefunctions saved via export_wf """
        self.beta_all = np.loadtxt(fn + '.betas')
        self.wf_all = np.load(fn + '.npy')
        
    def export_phase_space_svd_sparse_beta(self, fn = 'phase_space_svd_sparse_beta', bases_beta=None):
        """ assume all wavefunctions are saved in self.wf_all, as *row* vectors, carry out svd using basis states at beta = bases_beta. Then use the orthonormalized bases to expand all wavefunctions """

        if isNone(bases_beta):
            bases_beta = self.beta_all
        bases_idx = [ self.beta_all.searchsorted(beta) for beta in bases_beta ]
        actual_betas = self.beta_all[bases_idx]
        uu,dd,vv = la.svd(self.wf_all[bases_idx], full_matrices = False)

        header = gen_header('beta sval')
        with open_for_write(fn + '_sval.dat') as dat_sval:
            print('# two columns not related, i.e., sval not a fn of beta', file=dat_sval)
            print(header, file=dat_sval)
            for b,d in zip(actual_betas, dd):
                print('%g\t%g'%(b,d), file=dat_sval)

        header = gen_header('beta total_w coef1 coef2 ...')
        with open_for_write(fn + '.dat') as dat:
            print(header, file=dat)
            for beta, u in zip(self.beta_all, self.wf_all):
                print('\rbeta = %g                '%beta, end='')
                w,uproj,coef = proj_ortho_basis(u, vv)
                print('%g\t%g\t%s'%(beta, w, '\t'.join(['%g'%d for d in coef])), file=dat)


    def export_proj(self, bases, fn='var_proj'):
        """ assume all wavefuncntions are saved in self.wf_all, as *row* vectors, project them onto a given orthonormal bases """
        header = gen_header('beta total_w coef1 coef2 ...')
        with open_for_write(fn + '.dat') as dat:
            print(header, file=dat)
            for beta, u in zip(self.beta_all, self.wf_all):
                print('\rbeta = %g                '%beta, end='')
                w,uproj,coef = proj_ortho_basis(u, bases)
                print('%g\t%g\t%s'%(beta, w, '\t'.join(['%g'%d for d in coef])), file=dat)
                
                
