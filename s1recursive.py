# Time-stamp: <2018-02-01 13:08:49 zshuang>
# testing recursive sol of S=1 Heisenberg
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
SX = (SP + SM)/2
ISY = (SP - SM)/2
# S kron S
SS = 0.5 * sparse.kron(SP,SM) + 0.5 * sparse.kron(SM,SP) + sparse.kron(SZ,SZ)
SS_sqr = SS.dot(SS)

# dense version of S dot S
SSr = np.asarray(np.real(SS.todense()))

def apply_ss_1wf(wf, site):
    """ apply S_i dot S_i+1 to a single wf for i = site. 
    
    site = 0, 1, 2, ..., L-1

    wf must have the shape (3,3,3,3...) """

    nsites = len(wf.shape) # number of sites

    # shift site and site+1 to 0 and 1
    ww = np.moveaxis(wf, [site, (site + 1)%nsites], [0,1]).reshape(9, -1)
    res = np.dot(SSr, ww).reshape([3]*nsites)
    res = np.moveaxis(res, [0,1], [site, (site+1)%nsites])
    return res

def apply_H_1wf(wf):
    """ apply the Heisenberg H to a single wf """
    nsites = len(wf.shape)

    res = np.zeros_like(wf)
    for s in np.arange(nsites):
        #msg('s = ', s)
        res += apply_ss_1wf(wf, s)
    return res

def apply_ss_wfs(wfs, site):
    """ apply S_i dot S_i+1 to a set of wfs, for i = site

    site = 0, 1, 2, ..., L-1

    wfs = [wf1, wf2, wf3, ...], where each wf has shape (3,3,3,...)
    """

    nsites = len(wfs.shape) - 1 # number of sites
    nwf = wfs.shape[0] # number of wavefunctions

    res = np.moveaxis(wfs, 0, -1) # shift wf index to the last one
    res = np.moveaxis(res, [site, (site+1)%nsites], [0, 1]).reshape(9, -1) # move the relevant spins up front
    res = np.dot(SSr, res).reshape([*([3]*nsites), nwf])
    res = np.moveaxis(res, [0,1], [site, (site+1)%nsites])

    return np.moveaxis(res, -1,0) # move the wf index back to the first

def apply_H_wfs(wfs):
    """ apply the Heisenberg H to a set of wfs, see apply_ss_wfs() for wf specification """

    nsites = len(wfs.shape) - 1 # number of sites

    res = np.zeros_like(wfs)
    for s in np.arange(nsites):
        res += apply_ss_wfs(wfs, s)
    return res

class Heis_recur:
    def __init__(self, nsites=10, wf_init = None):
        self.nsites = nsites
        self.wf_init = wf_init
        if isNone(wf_init):
            wf_init = np.ones([3]*nsites) # initial wf
            self.wf_init = wf_init / la.norm(wf_init)
        self.wf_current = self.wf_init

    def gen_var_space(self,include_wf_init = True):
        """generate variational space from self.wf_current

        if include_wf_init == True, then also include the initial wf in
        the variational space
        """

        # dimension of the variational space
        dim_var = self.nsites + 1 # wf_current, and s.s |wf_current>
        if include_wf_init:
            dim_var += 1 # wf_init

        res = np.zeros([dim_var, *([3]*self.nsites)], dtype=np.double)

        res[-1] = self.wf_current
        if include_wf_init:
            res[-2] = self.wf_init
            
        for s in np.arange(self.nsites):
            res[s] = apply_ss_1wf(self.wf_current,s)
            res[s] /= la.norm(res[s])

        res = res.reshape(dim_var, -1) # each row a wavefunction
        v,d,u = la.svd(res, full_matrices=False)

        # rows of u now represent an orthonormal set
        u = u.reshape([dim_var, *([3]*self.nsites)])
        return u,d,res
    def gen_var_H(self, u_var):
        """ generate projected Hamiltonian in the variational bases u.

        Each u[i] is of shape (3,3,3,..), and is one variational basis state (and they are orthonormal)

        """

        dim_var = u_var.shape[0] # variational dimension

        hu = apply_H_wfs(u_var)

        res = np.dot(hu.reshape(dim_var, -1), npext.dagger( u_var.reshape(dim_var, -1) ))


        return res

    def var_iter(self, include_wf_init = False):
        """ iterate the variational procedure """
        
        msg('generating variational space...')
        u_var,d,r = self.gen_var_space(include_wf_init)

        dim_var = u_var.shape[0]

        msg('generating variational H ...')
        h_var = self.gen_var_H(u_var)

        msg('diagonalizing variational H')
        eig,u = la.eigh(h_var)

        msg('computing new wf')
        wf_new = np.dot(u[:,0], u_var.reshape(dim_var, -1)).reshape([3]*self.nsites)

        #msg('computing exact energy')
        #erg = np.sum(wf_new.conj() * apply_H_1wf(wf_new))

        self.wf_current = wf_new

        return wf_new, eig[0]

    def run_var_iter(self, include_wf_init = False, stop=1e-3):
        """ run the iteration until *percentage* of energy change falls below stop """

        erg_old = self.var_iter(include_wf_init)[-1]

        it = 1

        while True:
            it += 1
            print('Iteration %d:     current erg is %g'%(it, erg_old))
            erg_new = self.var_iter(include_wf_init)[-1]

            if np.abs((erg_old - erg_new)/erg_old) < stop:
                break

            erg_old = erg_new
        
        
class Heis_recur_1(Heis_recur):
    def __init__(self, nsites=10, wf_init = None):
        super().__init__(nsites = nsites, wf_init = wf_init)

    def gen_var_space(self, *param):
        """ generate the variational space. It will only consist of two states: wf_current, the component orthogonal to it after acting H on it. """

        h_wf = apply_H_1wf(self.wf_current)
        erg_current = np.sum(self.wf_current.conj() * h_wf) # <wf | H | wf>

        wf_perp = h_wf - erg_current * self.wf_current

        wf_perp /= la.norm(wf_perp)
        return np.asarray([self.wf_current, wf_perp]), None, None
