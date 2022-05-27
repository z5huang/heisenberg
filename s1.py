# Time-stamp: <2018-01-12 19:08:07 zshuang>
# S=1 Heisenberg

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

# Spin-1 matrices
SP = sparse.csr_matrix( np.sqrt(2) * np.diag(np.array([1,1],dtype=np.double), 1) )
#SM = sparse.csr_matrix( npext.dagger(SP) )
SM = SP.H
SZ = sparse.csr_matrix( np.diag(np.array([1,0,-1], dtype=np.double)) )
# S kron S
SS = 0.5 * sparse.kron(SP,SM) + 0.5 * sparse.kron(SM,SP) + sparse.kron(SZ,SZ)
SS_sqr = SS.dot(SS)

def s1ex_op(sep=0):
    """ operator for exchanging two spin-1s with sep spins in between """

    id_sep = sparse.identity(3**sep)

    res = 0
    for i,j in it.product(range(3), range(3)):
        sij = np.zeros((3,3))
        sij[i,j] = 1
        sij = sparse.csr_matrix(sij)
        res += functools.reduce(sparse.kron, (sij, id_sep, sij.T))
    return res

def s1ex(wf,i,j,nsites=None):
    """ given a wavefunction wf, exchange the ith and the jth sites """
    shape = wf.shape
    if nsites == None:
        nsites = int(round( math.log(np.prod(shape)) / math.log(3)))
    return np.swapaxes(wf.reshape([3]*nsites), i,j).reshape(shape)

def vortex_op(lev = 1):
    """ a level lev vortex operator """

    # size of the vortex
    vsize = 2*lev

    res = sparse.identity(3**vsize)
    for beg in range(0,lev, 2):
        id_beg = sparse.identity(3**beg)
        sep = vsize - 2*(beg + 1)
        msg('beg = ', beg, ' sep = ', sep)
        s_ex = s1ex_op(sep)
        res = res.dot( kron_reduce(id_beg, s_ex, id_beg) )
    return res

def vortex(wf,start=0,lev=1,nsites=None):
    """ apply a vortex to a wf """
    shape = wf.shape
    if nsites == None:
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
    if nsites == None:
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
    
def gen_translation(wf, nsites=None):
    """ generate translation copies of a wavefunction """
    shape = wf.shape
    if nsites == None:
        nsites = int(round( math.log(np.prod(shape)) / math.log(3) ))

    ww = wf.reshape([3]*nsites)
    #msg(ww)
    res = np.zeros([nsites, *[3]*nsites], dtype=wf.dtype)
    for t,_ in enumerate(it.repeat(None, nsites)):
        ww = np.rollaxis(ww, -1)
        res[t] = ww
        #msg(la.norm(res.flatten()))
    return res.reshape(nsites, -1)
    

def int2spin(n, nsites=0):
    """ convert an integer to a spin 1 configuration
    
    if nsites > 0, pad -1 to the left for an nsite configuration
    """
    repr = np.base_repr(n,3).zfill(nsites)
    if (nsites > 0) and (len(repr) > nsites):
        print('warning: len(repr) = %d, nsites = %d, there are more spins than sites. Returning the rightmost %d spins'%(len(repr),nsites,nsites))
        repr = repr[-nsites:]
    return np.array(list(repr),dtype=int) - 1

def is_SO(n,nsites=12):
    """ check if configuration n is string-ordered """
    s = int2spin(n,nsites)
    ss = str(s[s != 0])
    return not ( (ss.find('1 1') >= 0) or (ss.find('-1 -1') >= 0) )
    

def spin2int(s):
    """ convert a spin 1 configuration to an integer """
    base3 = s+1 # from {-1,0,1} to {0,1,2}
    srep =  np.array2string(base3,separator='')[1:-1]
    return int(srep, 3)

def spins2ints(spins):
    """convert an nparray (of shape M x N) representing M leng-N spin configurations to M integers
    """

    base3 = spins + 1
    basis = 3**(np.arange(spins.shape[1])[::-1]) # [3^(N-1), 3^(N-2), ..., 3^2, 3, 1]
    return np.sum(base3 * basis, axis=1)

def inv(sint, nsites=10):
    """ invert spin configuration represented by sint, an integer """
    s = int2spin(sint, nsites=nsites)
    sinv = s[::-1]
    return spin2int(sinv)

def kron_reduce(*ops):
    """ take the kron product of all inputing operators, _in order_ """
    return functools.reduce(sparse.kron, ops)

def count_updn(sint, nsites=10):
    """ count the number of up and down spins of a configuration """
    base3 = np.base_repr(sint,3)
    l = len(base3)
    if l < nsites:
        base3 = '0'*(nsites - l) + base3
    elif l > nsites:
        base3 = base3[-nsites:]
        msg('This shouldn\'t have happened, but I got more than %d spins'%nsites)
    nup = base3.count('2')
    ndn = base3.count('0')
    return (nup,ndn)

def gen_sz0_indices_old(nsites=10):
    """ generate all integers representing total Sz singlets """

    nup,ndn = np.array(list(map(functools.partial(count_updn,nsites=nsites), range(3**nsites)))).T

    res = np.arange(3**nsites)[nup - ndn == 0]
    return res

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
    
    


erg_abs = lambda f, s=0 : np.sum(np.abs(f))
erg_count1 = lambda f, s=0 : -np.abs(f).tolist().count(1)
erg_1over = lambda f, s=0 : np.sum(1/np.abs(f))
erg_sqrt = lambda f, s=0 : np.sum(np.sqrt( np.abs(f) ))
erg_diff1_sqr = lambda f, s=0 : np.sum((np.abs(f) - 1)**2)
erg_maxflux = lambda f, s=0 : np.max(np.abs(f))

def erg_no0_abs(f, spin=[]):
    ff = f[spin != 0]
    return erg_abs(ff)
def erg_coulomb(flux,spin):
    s1 = np.roll(spin,1)
    s2 = spin
    return np.sum(s1 * flux * s2)
    
def flux(spin, ergf=erg_abs):
    """compute the 'electrostatic' flux by minimizing 'electrostatic energy' 


    Think of spin 1 as charge 2, and spin -1 as charge
    -2. 'electrostatic' flux is defined as going from charge +1 toward
    charge -1. We will compute the number of flux lines between
    neighboring spins, where rightward is positive direction. Since
    charge only gives flux difference, there is an overall flux
    ambiguity, f0, which we assume to be the flux over the boundary,
    i.e., the flux going into the first spin from the left. 

    Consider, for example, a spin configuration and the corresponding
    flux configurations:

    S =      [    1     -1    -1     0     0     1 ]
    F = f0 + [ 0     2      0    -2    -2    -2    ]

    The Berry phase over a bond i is F[i] times pi, i.e., the flux F[i]
    is the winding number. From the Berry phase physics, we know that f0
    must be an odd integer. Physically, one also expects that min[F] >=
    -L/2 and max[F] <= L/2, where L is the chain length. This is because
    the flux line "prefers" to connect between spins over the shorter of
    the two paths (one through the 'bulk', the other across the
    'boundary'), and only L/2 sites will be within the "shorter" path;
    since there are at most L flux lines anywhere over the air (a flux
    line must emanate from a + charge, and end on a - charge, since
    there are at most 2L + charges, there are at most L flux lines),
    at most only half of them will belong to the shorter path.

    The physical picture of "shorter path preference" can be captured by
    minimizing certain "flux energy", over all possible f0 within the
    aforementioned bounds. I have tried three types:

    (1) E = sum_i F[i]^2

    (2) E = sum_i abs(F[i])

    (3) E = - number of abs(F[i])=1 (i.e., maximising the number of bond flux = +/- 1

    Upon numerical test, (1) is not good, e.g., for the L=14
    configuration [1,1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,-1]. Essentially none
    of them good for all cases
    """

    dflux = 2*np.roll(np.cumsum(spin),1)

    f0min = -np.min(dflux) - len(spin)//2
    f0max = -np.max(dflux) + len(spin)//2

    f0 = []
    erg = []
    flux = []
    for f in range(f0min,f0max+1):
        if f%2 == 1:
            f0.append(f)
            ff = f + dflux
            flux.append(ff)
            erg.append( ergf(ff, spin) )
    s = np.argsort(erg)
    return ( np.asarray(f0)[s],
             np.asarray(flux)[s],
             np.asarray(erg)[s], )

def gen_flux(spin):
    """ return dflux and all possible f0"""
    dflux = 2*np.roll(np.cumsum(spin),1)

    f0min = -np.min(dflux) - len(spin)//2
    f0max = -np.max(dflux) + len(spin)//2

    f0 = np.arange(f0min,f0max+1)
    f0 = f0[f0%2 == 1]

    return (f0,dflux)
    

class Heisenberg:
    def __init__(self,nsites=10):
        self.nsites = nsites
        self.hbulk = self.memo_hbulk()
        self.blink = self.memo_blink()
        #self.sz0idx = self.sz0_indices()
        msg('Computing total sz=0 bases...')
        self.sz0idx = gen_sz0_indices(nsites)
    
    def memo_hbulk(self):
        """ bulk Hamiltonian without boundary link """
        
        msg('constructing bulk H for nsites = %d...'%self.nsites)
        res = sparse.csr_matrix((3**self.nsites,3**self.nsites),dtype=np.double)
        for i in range(self.nsites - 1):
            id1 = sparse.identity(3**i, dtype=np.double)
            id2 = sparse.identity(3**(self.nsites - 2 - i))
            res += kron_reduce(id1, SS, id2)
        msg('bulk H constructed')
        return res

    def memo_blink(self):
        """ memoize boundary link operators """
        
        idmid = sparse.identity(3**(self.nsites-2), dtype=np.double)
        spsm = kron_reduce(SP,idmid,SM)
        szsz = kron_reduce(SZ,idmid,SZ)
        return (spsm,szsz)
    def hk(self,k):
        """ Hamiltonian with twisted boundary phase k

        must run memo_hbulk and memo_blink first
        """
        spsm,szsz = self.blink
        eik = np.exp(1j * k)
        hlink = 0.5 * spsm * eik + 0.5 * (spsm*eik).H + szsz
        return self.hbulk + hlink

    def memo_wf(self,nk=50,kfrom=0,kto=2*np.pi, kaug = np.pi + np.pi * 0.1 * np.arange(-5.5, 6)):
        """ memoize wavefunctions over a k loop

        nk: number of k points
        """

        sidx = self.sz0idx

        all_k = np.linspace(kfrom, kto, nk, endpoint=None)
        if kaug != None:
            all_k = np.concatenate((all_k, kaug))
            all_k.sort()
        self.all_k = all_k

        res = np.zeros((len(all_k), len(sidx)), dtype=np.complex)
        for x,k in enumerate(all_k):
            print('\rComputing %d/%d            '%(x,nk),end='')
            hh = self.hk(k)
            eig,u = sla.eigsh(hh,1,which='SA')
            res[x] = u[sidx].T[0]

        self.wf = res
        self.kto = kto
        self.kfrom = kfrom

        return res

    def save_par(self, fn):
        with open(fn + '.par', 'w') as par:
            txt = """
nsites = %d
"""%(self.nsites)
            print(txt,file=par)

    def save_wf(self, fn='full_wf'):
        """ save the wavefunctions over a k loop """

        self.save_par(fn)
        np.savetxt(fn + '.index', self.sz0idx,'%d')
        ## NB: np apparently has a bug which prevents it from reading a complex-valued txt file using loadtxt
        np.savetxt(fn + '.k', self.all_k, '%g')
        np.save(fn + '.dat', self.wf) # saved to fn+'.dat.npy'

    def read_par(self, fn):
        with open(fn + '.par', 'r') as par:
            nsites = int(par.read().strip().split('=')[-1])
        return nsites

    def read_wf(self,fn='full_wf'):
        """ read saved files """
        wf = np.load(fn + '.dat.npy')
        sz0idx = np.loadtxt(fn + '.index')
        all_k = np.loadtxt(fn + '.k')

        par = self.read_par(fn)
        return (par, sz0idx, wf)

class H_inter(Heisenberg):
    """ Interpolation between Heisenberg and AKLT, set by beta. beta=1/3 is AKLT. |beta| < 1 is in the Haldane phase """
    def __init__(self,nsites=10, beta=1/3):
        self.beta = beta
        super().__init__(nsites = nsites)

    def memo_hbulk(self):
        """ bulk Hamiltonian without boundary link """
        msg('constructing bulk H for nsites = %d and beta = %g...'%(self.nsites, self.beta))
        res = sparse.csr_matrix((3**self.nsites,3**self.nsites),dtype=np.double)
        for i in range(self.nsites - 1):
            id1 = sparse.identity(3**i, dtype=np.double)
            id2 = sparse.identity(3**(self.nsites - 2 - i))
            res += kron_reduce(id1, (SS + self.beta * SS_sqr), id2)
        msg('bulk H constructed')
        return res

    def hk(self,k):
        """ Hamiltonian with twisted boundary phase k

        must run memo_hbulk and memo_blink first
        """
        spsm,szsz = self.blink
        eik = np.exp(1j * k)
        ss_link = 0.5 * spsm * eik + 0.5 * (spsm*eik).H + szsz
        hlink = ss_link + self.beta * ss_link.dot(ss_link)
        return self.hbulk + hlink

    def save_par(self, fn):
        par = {'nsites': self.nsites, 'beta': self.beta}
        with open(fn + '.par', 'w') as par_file:
            json.dump(par,par_file)
    def read_par(self, fn):
        with open(fn + '.par', 'r') as par_file:
            par = json.load(par_file)
        return par

    @classmethod
    def export_gap(cls,nsites=10, nk=50, beta_from = -1.1, beta_to = 1.11, dbeta=0.05, fn='h_inter_gap'):
        """check if the gap remains open in the entire k space. 

        This is in response to the following observation: for n = 12, at
        beta = -0.5, the first excited state at k=pi is orthogonal to
        the ground state at k=0, while the ground state at k=pi has
        finite overlap with the ground state at k=0. This must mean that
        between beta=-0.5 and AKLT (beta = 1/3), the gap over k has
        collapsed.
        """

        with open(fn + '.par', 'w') as par:
            txt = """
nsites = %d
nk = %d
beta_from = %g
beta_to = %g
dbeta = %g
"""%(nsites, nk, beta_from, beta_to, dbeta)
            print(txt, file=par)

        header = gen_header('beta k eig1 eig2')

        with open(fn + '.dat', gen_mode(fn + '.dat')) as dat:
            print(header, file=dat)
            
            for beta in np.arange(beta_from, beta_to, dbeta):
                model = cls(nsites = nsites, beta=beta)
                for k in np.linspace(0, 2*np.pi, nk, endpoint=None):
                    hh = model.hk(k)
                    eig,u = sla.eigsh(hh, k=2, which='SA')
                    txt = '%g\t%g\t%g\t%g'%(beta,k,*eig)
                    #print('\rbeta, k = %g, %g                 '%(beta,k), end='')
                    print(txt)
                    print(txt, file=dat)
                dat.write('\n')

class Heis_var(H_inter):
    """ heisenberg variational solution """
    def __init__(self, nsites=10):
        super().__init__(nsites=nsites, beta=1/3)
        self.h_aklt = self.memo_h_aklt()
        self.u_aklt = sla.eigsh(self.h_aklt, k=1, which='SA')[1]
    def memo_h_aklt(self):
        """ memoize the pbc hamiltonian """
        spsm,szsz = self.blink
        ss_link = 0.5*(spsm  + spsm.H) + szsz
        hlink = ss_link + self.beta * ss_link.dot(ss_link)
        res = self.hbulk + hlink
        return res
    def s1ex_op(self,i,j):
        """ operator for exchanging spins i and j """
        if i==j:
            return sparse.identity(3**self.nsites)
        if i > j:
            i,j = j,i
        id_pre = sparse.identity(3**i)
        id_post = sparse.identity(3**(self.nsites - j - 1))
        res = kron_reduce(id_pre, sparse.csr_matrix( s1ex_op(j-i-1) ), id_post)
        return res
    def u_ex_older(self,sep=0):
        """ from the aklt state, generate a translationally invariant single-vortex state, by exchanging spins with separation sep """
        res = 0
        for i in range(self.nsites):
            #msg('i = ', i)
            j = (i+1+sep)%self.nsites
            res += self.s1ex_op(i,j).dot(self.u_aklt)
        return res / la.norm(res)

    def u_ex_old(self, sep=0):
        u0 = self.s1ex_op(0,1+sep).dot(self.u_aklt).reshape([3]*self.nsites)
        #msg(u0)
        res = 0
        for _ in it.repeat(None,self.nsites):
            u0 = np.rollaxis(u0, -1)
            res += u0
            msg(la.norm(res.flatten()))
        res = res.flatten()
        return res / la.norm(res)

        #u0 = s1ex(self.u_aklt, 0, 1+sep, nsites=self.nsites)
        #return homogenize(u0, self.nsites)
    def u_ex(self, sep=0):
        """ Starting from the AKLT ground state, exchange two spins with sep spins in between, then translation-symmetrize it (homogenize) """
        u0 = s1ex(self.u_aklt, 0, 1+sep, nsites=self.nsites)
        return homogenize(u0, self.nsites)

    def u_ex_multi(self, *isites):
        """ carry out multiple nearest neighbor exchanges, with the first sites specified by isites """
        res = self.u_aklt.reshape([3]*self.nsites)
        
    def u_ex_nosum_older(self,sep=0):
        """ from the aklt state, generate a translationally invariant single-vortex state, by exchanging spins with separation sep """
        res = []
        for i in range(self.nsites):
            #msg('i = ', i)
            j = (i+1+sep)%self.nsites
            res.append((self.s1ex_op(i,j).dot(self.u_aklt)).flatten())
        return res
def proj_nonortho_basis(u, *ubasis):
    """ project the target state u to the space spanned by a set of non-orthogonal basis states ubasis """

    u = u.flatten()
    umat = np.asarray([ uu.flatten() for uu in ubasis ])

    # Ortho-normalize ubasis. Rows of vv form ortho-normal basis of the
    # umat space
    uu,dd,vv = la.svd(umat, full_matrices=False)
    coef = np.sum(u.conj() * vv, axis=-1)

    weight = np.sum(coef * coef.conj())
    uproj = np.sum(coef.reshape(-1,1) * vv, 0)
    uproj /= la.norm(uproj)
    return weight,uproj
    
def expand_vortex(heis_var, u_heis, sep_max = 3):
    u_aklt = heis_var.u_aklt.flatten()
    u_heis = u_heis.flatten()
    umat = [u_aklt]
    for sep in range(sep_max+1):
        u_ex_sep = heis_var.u_ex(sep).flatten()
        umat.append(u_ex_sep)
    umat = np.asarray(umat)
    uu,dd,vv = la.svd(umat, full_matrices = False)

    

    res = [ np.sum(u_heis * vv[i])**2 for i in range(sep_max+2) ]
    wf_var = np.sum(np.sum(u_heis * vv, axis=-1).reshape(-1,1) * vv, axis=0)
    wf_var /= la.norm(wf_var)
    return np.sqrt(np.sum(res)), res, wf_var, [uu,dd,vv]
                
    


class Hwf:
    """ class for manipulating the wavefunction of the S=1 Heisenberg model """
    @classmethod
    def read(cls,fn='n=12'):
        res = cls()
        res.read_wf(fn)
        return res
    
    def __init__(self,n=None,sconf=None,wf=None):
        self.n = n          # chain length
        self.sconf = sconf      # total sz=0 configurations
        self.wf = wf        # wavefunctions in the same order as sconf
        
    def save_par(self, fn):
        with open(fn + '.par', 'w') as par:
            txt = """
nsites = %d
"""%(self.n)
            print(txt,file=par)

    def save_wf(self, fn='full_wf'):
        """ save the wavefunctions over a k loop """

        self.save_par(fn)
        np.savetxt(fn + '.index', self.sz0idx,'%d')
        ## NB: np apparently has a bug which prevents it from reading a complex-valued txt file using loadtxt
        np.save(fn + '.dat', self.wf) # saved to fn+'.dat.npy'
        np.savetxt(fn + '.k', self.all_k, '%g')

    def read_par(self, fn):
        with open(fn + '.par', 'r') as par:
            nsites = int(par.read().strip().split('=')[-1])
        self.n = nsites

    def read_wf(self,fn='n=12'):
        """ read saved files """
        self.wf = np.load(fn + '.dat.npy')
        self.all_k = np.loadtxt(fn + '.k', dtype=np.double)
        self.sconf = np.loadtxt(fn + '.index',dtype=int)
        self.read_par(fn)
        self.sconf_list = list(self.sconf) # also save as python list for quick lookup
        self.sort_wf()
        self.fix_gauge()
        self.gen_s2idx()
        
    def sort_wf(self):
        """ sort wavefunctions and indices by decreasing weight """
        s = np.argsort(np.abs(self.wf[0]))[::-1]
        self.wf = self.wf[:,s]
        self.sconf = self.sconf[s]
        self.sconf_list = list(self.sconf) # also save as python list for quick lookup

    def fix_gauge(self):
        """ fix the gauge of all wavefunctions with reference to the highest weight component (either [-1,1,-1,1,...] or [1,-1,1,-1,...] 

        Note: run sort_wf() first
        """
        phases = self.wf[:,0] / np.abs(self.wf[:,0])
        self.wf = (self.wf.T / phases.T).T
        
    def gen_s2idx(self):
        """ generate the spin_configuration-to-index map """
        s2idx = np.zeros(np.max(self.sconf) + 1, dtype=int)
        for x,s in enumerate(self.sconf):
            s2idx[s] = x
        self.s2idx = s2idx

    def gen_trans_conf_with_idx(self,sconf):
        """ generate unique configurations related by translation """

        conf = [sconf]
        conf_idx = [self.sconf_list.index(sconf)]

        spin = int2spin(sconf,self.n)
        # translate by 2, 4, etc., then by 1, 3, etc.
        tr = list(range(2,self.n,2)) + list(range(1,self.n,2))
        for x in tr:
            ss = spin2int(np.roll(spin, x))
            if conf.count(ss) == 0: # only append non-repetitive configurations
                conf.append(ss)
                conf_idx.append(self.sconf_list.index(ss))

        return (conf,conf_idx)

    def get_spin_repr(self,sconf):
        """ get the integer and nparray repr of a spin configuration 

        sconf is either an integer, or a list/nparray
        """
        if isinstance(sconf, collections.Iterable): # literal configuration
            spin = np.asarray(sconf)
            sint = spin2int(spin)
        else:
            spin = int2spin(sconf, self.n)
            sint = sconf
        return (sint,spin)
    def gen_trans_conf(self,sconf):
        """generate unique configurations related by translation. 

        Also checks if sconf is ISE
        """

        sint,spin = self.get_spin_repr(sconf)

        conf = [sint]
        conf_spin = [spin]

        #spin = int2spin(sconf,self.n)
        # translate by 2, 4, etc., then by 1, 3, etc.
        tr = list(range(2,self.n,2)) + list(range(1,self.n,2))
        for x in tr:
            si,sp = self.get_spin_repr(np.roll(spin, x))
            if conf.count(si) == 0: # only append non-repetitive configurations
                conf.append(si)
                conf_spin.append(sp)
        return (conf,conf_spin)
    def iter_sconf(self):
        """Iterator of the spin configurations

        Returns an iterator where translationally related configurations
        are clusterred together. If the configuration is not Inversion
        Self Equivalent (ISE), then also include a batch of inversion
        partners after the batch of translation generated configurations

        ISE: "Inversion Self Equivalent", a configuration related to its
        inversion conjugate by an _even_ number of elementary translations

        ISE0: if a configuration is reflection symmetric wrt an s=0 site. 

        Empirically, both ISE and ISE0 configurations have zero
        wavefunction weight at k=pi, and therefore (s, inv(s)) cannot be
        assigned an azimuthal angle at k=pi.

        emits (conf, is_ISE, is_ISE0) upon each yield, where conf
        consists of translation and inversion related configurations
        """

        conf_generated = np.zeros(np.max(self.sconf)+1,dtype=bool)
        #for x,s in enumerate(self.sconf):
        def _is_ISE(c):
            return np.array_equal(c, c[::-1])
        for s in self.sconf:
            if conf_generated[s] == False: # haven't generated it yet
                #print('x = ', x)
                conf, conf_spin = self.gen_trans_conf(s)
                is_ISE = ([ _is_ISE(c) for c in conf_spin ].count(True) > 0)
                is_ISE0 = ([ (c[0] == 0 and _is_ISE(c[1:])) for c in conf_spin ].count(True) > 0)

                sinv = inv(s,self.n)
                #is_ISE = conf[:len(conf)//2].count(sinv)
                if conf.count(sinv) == 0:
                    conf_inv = self.gen_trans_conf(sinv)[0]
                    conf += conf_inv
                yield((conf, is_ISE, is_ISE0))
                conf_generated[conf] = True
                        
    ################################################################
    def print_info(self,from_conf=0, size=72, kidx=0):
        """ print some information of the wavefunctions 
        
        print from configuration index 'from_conf' to 'from_conf + size'
        """

        spec='conf_idx wf_weight[kidx] s-conf'
        msg('# ', spec)
        for x,n in enumerate(self.sconf[from_conf:from_conf + size]):
            print('% 8d:'%(x + from_conf), np.abs(self.wf[kidx, x+from_conf]), ':', int2spin(n,self.n))
        msg('# ', spec)

    def pair_info(self,sconf):
        """ compute information on a spin configuration sconf, and its inversion pair

        NB: must sort_wf() and fix_gauge() first
        """

        sint1,spin1 = self.get_spin_repr(sconf)

        spin2 = spin1[::-1]
        sint2 = spin2int(spin2)
            
        if sint1 == sint2:
            msg('self dual, will return 0')
            return 0

        wf1 = self.wf[:,self.s2idx[sint1]]
        wf2 = self.wf[:,self.s2idx[sint2]]

        phases = wf2 / wf1
        phases2 = np.roll(phases,-1)
        dphi = np.angle(phases2 / phases) # phase advances

        wfabs = np.abs(wf1)

        res = {
            'wfmax' : np.max(wfabs),
            'wfmin' : np.min(wfabs),
            'wfmean': np.mean(wfabs),
            'wfstd' : np.std(wfabs),
            'dphi_max' : np.max(dphi),
            'dphi_min' : np.min(dphi),
            'dphi_mean': np.mean(dphi),
            'dphi_std' : np.std(dphi),
            'winding'  : np.sum(dphi) / (2*np.pi)
            }
        return res

    def print_wf(self, sconf, kidx=0):
        """ for each configuration in sconf, print the configuration and its wf at kidx """
        for s in sconf:
            idx = self.sconf_list.index(s)
            spin = int2spin(s,self.n)
            wf = self.wf[kidx,idx]
            print('%5d'%idx, ':', spin, ' : ', '%g'%np.abs(wf), ' , ', '%g'%(np.angle(wf)/np.pi), ' pi')
            
    def export_1wf(self, sconf):
        """ export the azimuth of an inversion pair, sconf and inv(sconf) """
        si1,sp1 = self.get_spin_repr(sconf)
        si2 = inv(si1, self.n)

        wf1 = self.wf[:, self.s2idx[si1]]
        wf2 = self.wf[:, self.s2idx[si2]]

        np.savetxt('tmp_phi.dat', np.asarray((self.all_k, np.angle(wf2/wf1))).T)
        np.savetxt('tmp_rho.dat', np.asarray((self.all_k, np.abs(wf1))).T)

    def gen_df(self, fn=None):
        """generate a DataFrame. Write to fn in python pickle format if fn is provided.

        Resulting df will be indexed by sint

        - sint: int repr of a spin configuration

        - sint_repr: the corresponding representative spin configuration.
        Configurations related by translation and inversion are equivalent by sint_repr

        - spin: the litera spin configuration

        - w: the computed winding number

        - possible_f0: a list of possible f0s

        - is_SO: if a configuration is string ordered

        - is_ISE: if a configuration is Inversion Self Equiv

        - wf_max, wf_min, wf_mean, wf_std: statistis of wf amplitude over the k-loop

        - dphi_max dphi_min dphi_mean dphi_std: statistics of the azimuthal angle 'velocity' over the k-loop
        """
        
        #res = pd.DataFrame(columns = ['sint_repr', 'spin', 'w',
        #                                  'f0', 'dflux',
        #                                  'is_SO', 'is_ISE',
        #                                  'wf_max', 'wf_min', 'wf_mean', 'wf_std',
        #                                  'dphi_max', 'dphi_min', 'dphi_mean', 'dphi_std' ])
        records = []

        total_written = 1
        total = self.sconf.shape[0]
        all_s = []
        for sconfs, is_ISE, is_ISE0 in self.iter_sconf():
            s_repr = sconfs[0]
            is_so = is_SO(s_repr,self.n)
            wf_repr = self.wf[:,self.s2idx[s_repr]]
            wf_abs = np.abs(wf_repr)
            wf_max = np.max(wf_abs)
            wf_min = np.min(wf_abs)
            wf_mean = np.mean(wf_abs)
            wf_std = np.std(wf_abs)

            for s in sconfs:
                print('\rWriting %d / %d         '%(total_written, total),end='')
                total_written += 1
                all_s.append(s)

                    
                idx = self.s2idx[s]
                wf1 = self.wf[:,idx]

                sinv = inv(s, self.n)
                idx_inv = self.s2idx[sinv]
                rwf = self.wf[:,idx_inv] / wf1 # relative wf
                dphi = np.angle(np.roll(rwf,-1) / rwf)
                dphi_max = np.max(dphi)
                dphi_min = np.min(dphi)
                dphi_mean = np.mean(dphi)
                dphi_std = np.std(dphi)
                w = int(round(np.sum(dphi)/(2*np.pi)))

                sp = int2spin(s,self.n)
                f0, dflux = gen_flux(sp)

                #res.loc[s] = [s_repr, sp, w,
                #                  f0, dflux,
                #                  is_so, is_ISE,
                #                  wf_max, wf_min, wf_mean, wf_std,
                #                  dphi_max, dphi_min, dphi_mean, dphi_std]
                records.append( [ s_repr, sp, w,
                                  f0, dflux,
                                  is_so, is_ISE or is_ISE0,
                                  wf_max, wf_min, wf_mean, wf_std,
                                  dphi_max, dphi_min, dphi_mean, dphi_std]
                            )
        res = pd.DataFrame(records,
                    columns = ['sint_repr', 'spin', 'w',
                               'f0', 'dflux',
                               'is_SO', 'is_ISE',
                               'wf_max', 'wf_min', 'wf_mean', 'wf_std',
                               'dphi_max', 'dphi_min', 'dphi_mean', 'dphi_std' ],
                    index = all_s)
        if fn != None:
            res.to_pickle(fn + '.pkl')
        return res #,records
                

    def digest(self,fn='digest-n=12'):
        """ digest wavefunction information and generate a report 

        - For each spin configuration, compute
            - if it is string-ordered
            - wfmax
            - wfmin
            - wfmean
            - wfstd
            - dphi_max
            - dphi_min
            - dphi_std
            - winding (must be integer but can be even. NB: gauge fixed to [-1,1,-1,1,...] being >0
            - If it is "Inversion Self Equivalent" (ISE), i.e., related to its inversion conjugate by inversion

        - For each inversion pair (non-ISE), compute:
            - dphi_max
            - dphi_min
            - dphi_std
            - winding (must be odd)
        """

        label = 'sconf_int sconf is_SO is_ISE w_pair w_sqrt w_count1 w_abs len(uniq[w]) pair_dphi_max pair_dphi_min pair_dphi_mean pair_dphi_std erg_sqrt gap_sqrt erg_count1 gap_count1 erg_abs gap_abs wfmax wfmin wfmean wfstd w_self dphi_max dphi_min dphi_mean dphi_std'

        digits = len(str(np.max(self.sconf)))
        digits = '%%%dd'%digits

        with open(fn + '.dat', gen_mode(fn + '.dat')) as dat:
            header = gen_header(label)
            print(header, file=dat)

            total_written = 1
            total = self.sconf.shape[0]
            for sconfs,is_ISE, is_ISE0 in self.iter_sconf():
                dat.write('#\n') # starting a new group

                s_repr = sconfs[0]
                is_so = is_SO(s_repr,self.n)
                wf_repr = self.wf[:,self.s2idx[s_repr]]
                wfabs = np.abs(wf_repr)
                wfmax = np.max(wfabs)
                wfmin = np.min(wfabs)
                wfmean = np.mean(wfabs)
                wfstd = np.std(wfabs)

                for s in sconfs:
                    print('\rWriting %d / %d         '%(total_written, total),end='')
                    total_written += 1
                    
                    idx = self.s2idx[s]
                    wf1 = self.wf[:,idx]
                    wf2 = np.roll(wf1, -1)

                    wfdphi = np.angle(wf2 / wf1)
                    dphi_max = np.max(wfdphi)
                    dphi_min = np.min(wfdphi)
                    dphi_mean = np.mean(wfdphi)
                    dphi_std = np.std(wfdphi)
                    w_self = np.sum(wfdphi)/(2*np.pi)

                    sinv = inv(s, self.n)
                    idx_inv = self.s2idx[sinv]
                    rwf = self.wf[:,idx_inv] / wf1 # relative wf
                    pair_dphi = np.angle(np.roll(rwf,-1) / rwf)
                    pair_dphi_max = np.max(pair_dphi)
                    pair_dphi_min = np.min(pair_dphi)
                    pair_dphi_mean = np.mean(pair_dphi)
                    pair_dphi_std = np.std(pair_dphi)
                    w_pair = np.sum(pair_dphi)/(2*np.pi)

                    # flux() may return empty result, presumably because
                    # there is no odd integer between [f0min, f0max]
                    try:
                        f_res = flux(int2spin(sinv, self.n), ergf=erg_sqrt)
                        w_sqrt = f_res[1][0][0]
                        e_sqrt = f_res[2][0]
                        if len(f_res[2]) > 1:
                            g_sqrt = f_res[2][1] - f_res[2][0]
                        else:
                            g_sqrt = -1 # negative gap means only one flux arrangement is possible
                        f_res = flux(int2spin(sinv, self.n), ergf=erg_count1)
                        w_count1 = f_res[1][0][0]
                        e_count1 = f_res[2][0]
                        if len(f_res[2]) > 1:
                            g_count1 = f_res[2][1] - f_res[2][0]
                        else:
                            g_count1 = -1

                        f_res = flux(int2spin(sinv, self.n), ergf=erg_no0_abs)
                        w_abs = f_res[1][0][0]
                        e_abs = f_res[2][0]
                        if len(f_res[2]) > 1:
                            g_abs = f_res[2][1] - f_res[2][0]
                        else:
                            g_abs = -1

                    except IndexError:
                        w_sqrt = w_count1 = w_abs = 0
                        e_sqrt = e_count1 = e_abs = 0
                        g_sqrt = g_count1 = g_abs = -2
                        msg(int2spin(s, self.n), ': No result returned from flux()')

                    nw = len(np.unique([round(w_pair), w_sqrt,w_count1,w_abs]))

                    fmt = [
                          [digits,  s,
                        ],['%s',    int2spin(s,self.n),
                        ],['%d',    is_so,
                        ],['%d',    is_ISE,
                        ],['% .0f', w_pair,
                        ],['% .0f', w_sqrt,
                        ],['% .0f', w_count1,
                        ],['% .0f', w_abs,
                        ],['%d',    nw,
                        ],['%+g',   pair_dphi_max,
                        ],['%+g',   pair_dphi_min,
                        ],['%+g',   pair_dphi_mean,
                        ],['%+g',   pair_dphi_std,
                        ],['%g',    e_sqrt,
                        ],['%g',    g_sqrt,
                        ],['%g',    e_count1,
                        ],['%g',    g_count1,
                        ],['%g',    e_abs,
                        ],['%g',    g_abs,
                        ],['%g',    wfmax,
                        ],['%g',    wfmin,
                        ],['%g',    wfmean,
                        ],['%g',    wfstd,
                        ],['% .0f', w_self,
                        ],['%+g',   dphi_max,
                        ],['%+g',   dphi_min,
                        ],['%+g',   dphi_mean,
                        ],['%+g',   dphi_std,
                        ],]
                    for s,v in fmt:
                        print(s%v, '\t', file=dat, end='')
                    dat.write('\n')
                        
    def export_proj_0_pi(self,fn='proj_0pi'):
        """ project all wavefunctions |\psi(k)> to a 2-dim basis of { |\psi(0)>, |\psi(pi)> } """

        wf0 = self.wf[0].conj()
        wfpi = self.wf[self.wf.shape[0]//2].conj()

        header = gen_header('k abs(psi_0) abs(psi_pi) arg(psi_0) arg(psi_pi) theta phi bx by bz')

        with open(fn+'.dat', gen_mode(fn+'.dat')) as dat:
            print(header, file=dat)
            for n,k in enumerate(self.all_k):
                wf = self.wf[n]
                proj0 = np.dot(wf0, wf)
                phase0 = proj0 / np.abs(proj0)
                projpi = np.dot(wfpi, wf)
                phi = np.angle(projpi / proj0)
                theta = 2*np.math.atan(np.abs(projpi / proj0))

                ss = np.sin((theta,phi))
                cc = np.cos((theta,phi))
                bx = ss[0] * cc[1]
                by = ss[0] * ss[1]
                bz = cc[0]

                print(('%g\t'*10)%(k,
                                   *np.abs([proj0, projpi]),
                                   *np.angle([proj0, projpi]),
                                   theta, phi, bx, by, bz), file=dat)

                
            
class Hwf_inter(Hwf):
    def read_par(self,fn):
        with open(fn + '.par', 'r') as par_file:
            par = json.load(par_file)
        self.n = int(par['nsites'])
        self.beta = par['beta']
    def save_par(self,fn):
        par = {'nsites': n, 'beta': b}
        with open(fn + '.par', 'w') as par_file:
            json.dump(par, par_file)
################################################################
## for DataFrame manipulations
class Hdf():
    def __init__(self, df_raw):
        self.df_raw = df_raw
    
        # do some clean up
        wf_nonzero = df_raw.wf_min > 1e-12    # only look at nonzero wf
        srepr = df_raw.index == df_raw.sint_repr  # only look at representative configurations
    
        self.df_repr = df_raw[srepr]

        self.df = df_raw[ wf_nonzero & srepr ]

    def check_ISE_wfmin(self, tol = 1e-12):
        """ return a sub df where ISE and wfmin give different result

        i.e., is_ISE != (wf_min < tol)
        """
        wf_zero = self.df_repr.wf_min < tol
        res = self.df_repr[wf_zero != self.df_repr.is_ISE]
        print('%d / %d representative configurations with inconsistent wf_min and is_ISE '%(len(res),len(self.df)))
        return res

    @classmethod
    def read(cls,fn='n=10'):
        df_raw = pd.read_pickle(fn + '.pkl')
        return cls(df_raw)

    def check_winding(self):

        flux = np.vstack(self.df.dflux - self.df.w)
        spin = np.vstack(self.df.spin)

        sint_tr = [ spins2ints(np.roll(spin, -d, axis=1)) for d in range(0, spin.shape[1]) ]
        w_all = np.vstack([ self.df_raw.loc[sint].w for sint in sint_tr ]).T

        res = np.all( ((w_all + flux) == 0), axis=1 )
        c = list(res).count(False)
        
        print('%d / %d representative configurations (wf_min > 0) with inconsistent winding vs flux '%(c,len(self.df)))
        return self.df[res == False]

    def check_winding_s(self, srepr):
        """check if the winding number over different links is indeed

        given by a flux arrangement
        """

        rec = self.df.loc[srepr]
        flux = -rec.w + rec.dflux
        #msg('flux = ', flux)
        #msg(srepr, ':', rec.spin)
        n = len(rec.spin)

        w_all = [rec.w]
        for d in range(1,n):
            sp = np.roll(rec.spin,-d)
            si = spin2int(sp)
            #msg(si, ':', sp)
            w_all.append(self.df_raw.loc[si,'w'])
        return (flux, w_all, np.array_equal(-flux, w_all))
        #return np.array_equal(-flux, w_all)

    def compute_winding_s(self,srepr,ergf):
        """ use ergf to sort all possible f0, and return the lowest two f0

        returns [w1, w2, gap]

        NB: 

        - returns gap == -1 if there is 0 or 1 f0

        - returns w1 = w2 = 0 if there is 0 f0
        
        """

        all_f0,dflux = self.df.loc[srepr, ['f0', 'dflux']]

        if len(all_f0) == 0:
            msg('this should not happen')
            return [0,0,-1]
        if len(all_f0) == 1:
            return [all_f0[0], all_f0[0], -1]

        ergs = np.asarray([ ergf(f0 + self.df.loc[srepr].dflux) for f0 in all_f0 ])
        idx_sorted = np.argsort(ergs)
        ergs_sorted = ergs[idx_sorted]
        return [*all_f0[idx_sorted[:2]], ergs_sorted[1] - ergs_sorted[0]]
    def compute_winding_old(self, ergf):
        """ use ergf to compute the winding of all srepr """

        res = np.vstack([ self.compute_winding_s(srepr, ergf) for srepr in self.df.index ])

        idx_degen = res[:,-1] == 0
        n_degen = idx_degen.tolist().count(True)

        idx_pred1 = (self.df.w + res[:,0]) == 0 # correctly predicted by lowest erg
        idx_pred2 = (self.df.w + res[:,1]) == 0 # correctly predicted by 2nd lowest erg
        idx_pred2_degen = idx_pred2 & idx_degen # correctly predicted by a degenerate erg

        n_pred1, n_pred2, n_pred2_degen = [ list(idx).count(True) for idx in (idx_pred1, idx_pred2, idx_pred2_degen) ]

        print("""
Total inequiv spin configurations: %d
Degenerate lowest ergs :           %d
Predicted by w1:                   %d
Predicted by w2:                   %d
Predicted by w1 + w2:              %d
Predicted by degenerate w2:        %d
Predicted total with degen:        %d
Failed to predict:                 %d"""%(len(self.df), n_degen, n_pred1, n_pred2, n_pred1 + n_pred2, n_pred2_degen, n_pred1 + n_pred2_degen, len(self.df) - n_pred1 - n_pred2_degen))
        return np.vstack(res)

    def compute_winding(self, ergf = None):
        """vectorized version for estimating the winding of all representative spin configurations by minimizing ergf.

        For different f0 flux, f0+dflux generates a possible flux
        configuration, F[f0]. ergf will assign an 'energy' or 'cost' to
        F[f0] => E[f0]. The "true" f0 is supposed to minimize this cost
        function.

        Physically, for different spin configurations, the number of
        possible f0 values are different, see description in 'def
        flux()'. This makes it difficult to vectorize the
        computation. Here, we will always use the same set of candidate
        f0 values, [-N//2, N//2], regardless of the spin (equivalently
        dflux) configurations.
        """

        if ergf == None:
            ergf = e_abs

        dflux = np.vstack(self.df.dflux)

        n = dflux.shape[-1] # number of spins
        f0 = np.arange(- n//2, n//2 + 1) # always use the same set of f0 candidates regardless of dflux
        f0 = f0[f0%2 == 1]

        f0expand = f0.reshape( (f0.shape[0], 1, 1) )   # to allow broadcast for f0 + dflux
        flux = f0expand + dflux

        ergs = ergf(flux) # of shape n_f0 x M_records
        idx_sorted = np.argsort(ergs, axis=0)  # sort along f0

        nrec = len(self.df) # number of inequiv spin configurations

        ## A note of multi-dimension indexing:
        ## M[ [x1,x2,x3,x4], [y1,y2,y3,y4] ] => [ M[x1,y1], M[x2,y2], M[x3,y3], M[x4,y4] ]
        ## Note that x and y arrays should have the same length. (otherwise numpy will attempt broadcasting)
        gap = ergs[idx_sorted[1], np.arange(nrec)] - ergs[idx_sorted[0], np.arange(nrec)]
        
        w1 = f0[idx_sorted[0]] + dflux[:,0]   # w with lowest energy
        w2 = f0[idx_sorted[1]] + dflux[:,0]   # w with second lowest energy
        
        idx_degen = (gap == 0)
        n_degen = list(idx_degen).count(True)
        idx_pred1 = (w1 + self.df.w) == 0
        idx_pred2 = (w2 + self.df.w) == 0
        idx_pred2_degen = idx_pred2 & idx_degen
        
        n_pred1, n_pred2, n_pred2_degen = [ list(idx).count(True) for idx in (idx_pred1, idx_pred2, idx_pred2_degen) ]

        print(
        """
Total inequiv spin configurations: %d
Degenerate lowest ergs :           %d
Predicted by w1:                   %d
Predicted by w2:                   %d
Predicted by w1 + w2:              %d
Predicted by degenerate w2:        %d
Predicted total with degen:        %d
Failed to predict:                 %d
"""%(nrec, n_degen, n_pred1, n_pred2, n_pred1 + n_pred2, n_pred2_degen, n_pred1 + n_pred2_degen, nrec - n_pred1 - n_pred2_degen))
        
        

def e_abs(flux):
    """ flux is of shape M1 x M2 x ... x N representing M1 x M2 x ... x len-N flux configurations """
    return np.sum( np.abs(flux), axis=-1 )

def add_reduce(*ops):
    """ take the kron product of all inputing operators, _in order_ """
    return functools.reduce(np.add, ops)

def e_count1(flux):
    return -np.sum( np.abs(flux) == 1, axis=-1 )
def e_test(flux, p=1):
    return np.sum( np.abs(flux)**p, axis=-1)
def e_test_a(**args):
    return functools.partial(e_test,**args)
def e_log(flux):
    return np.sum( np.log(np.abs(flux)), axis=-1 )
def e_flogf(flux):
    return np.sum( np.abs(flux) * np.log(np.abs(flux)), axis=-1 )

def e_pow(flux, b=[0.1]):
    """ power series of abs(flux) """

    aflux = np.abs(flux)

    res = 1.0*np.sum( aflux, axis=-1 )

    powers = np.arange(2, len(b)+2)
    for p,coef in zip(powers,b):
        res += coef * np.sum( aflux**p, axis=-1 )
    return res
def e_power_b(b):
    return functools.partial(e_pow, b=b)
        

################################################################
## for checking purposes
def h_heisenberg(nsites):
    """ Heisenberg Hamiltonian. 
    """
    #print('constructing Hamiltonian...')
    res = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        res += sparse.kron(id1, sparse.kron(SS, id2))

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    sx = 0.5 * (SP + SM)
    isy = 0.5 * (SP - SM)
    for op1,op2 in zip((sx,isy,SZ),(sx,-isy,SZ)):
        res += sparse.kron(op1, sparse.kron(id_mid, op2))
    #print('Hamlitonian constructed!')
    return res

def h_heisenberg_proj(nsites):
    """ Heisenberg Hamiltonian. Projected onto total sz=0
    """
    #print('constructing Hamiltonian...')

    sz0 = gen_sz0_indices(nsites)
    dim = sz0.shape[0]
    res = sparse.csr_matrix((dim,dim),dtype=np.double)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        hi = sparse.kron(id1, sparse.kron(SS, id2)).tocsr()
        res += hi[sz0][:,sz0]

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    sx = 0.5 * (SP + SM)
    isy = 0.5 * (SP - SM)
    for op1,op2 in zip((sx,isy,SZ),(sx,-isy,SZ)):
        hi = sparse.kron(op1, sparse.kron(id_mid, op2)).tocsr()
        res += hi[sz0][:,sz0]
    #print('Hamlitonian constructed!')
    return res

def h_aklt(nsites):
    res = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    pp = SS + SS.dot(SS)/3 + 2/3*sparse.identity(3**2)
    for i in range(nsites - 1):
        id1 = sparse.identity(3**i, dtype=np.double)
        id2 = sparse.identity(3**(nsites - 2 - i))
        res += sparse.kron(id1, sparse.kron(pp, id2))

    # boundary link
    id_mid = sparse.identity(3**(nsites - 2))
    sx = 0.5 * (SP + SM)
    isy = 0.5 * (SP - SM)
    s_s = sparse.csr_matrix((3**nsites,3**nsites),dtype=np.double)
    for op1,op2 in zip((sx,isy,SZ),(sx,-isy,SZ)):
        s_s += sparse.kron(op1, sparse.kron(id_mid, op2))
    p_p = s_s + s_s.dot(s_s)/3 + 2/3*sparse.identity(3**nsites)
    res += p_p
    return res

#### NB ####
## To obtain the eigenvalue, use sparse.linalg.eigsh(h, 1, which='SA')
## Here, 'SA' is important, it finds "the smallest eigenvalue inclusive of any negative sign"
