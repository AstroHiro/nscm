#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:43:48 2020

@author: hiroyasu
"""
import numpy as np
from classsncm import SNCM
from classncm_sn import NCM

np.random.seed(seed=8)
# enter your choice of CV-STEM sampling period
dt = 0.1
# specify upper and lower bounds of each state
xlims = np.array([-np.ones(2),np.ones(2)])*np.pi/5
#xlims = np.array([[-1,-2],[1,2]])*20*np.pi/180
plims = np.array([[2],[4]])
# specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
alims = np.array([0.1,30])
elims = np.array([1,3])
# name your NCM
fname = "SNCMmissileC"
# system constants
an = 0.000103*(180/np.pi)**3
bn = -0.009450*(180/np.pi)**2
cn = -0.169600*(180/np.pi)**1
dn = -0.034000*(180/np.pi)**1
am = 0.000215*(180/np.pi)**3
bm = -0.019500*(180/np.pi)**2
cm = 0.051000*(180/np.pi)**1
dm = -0.206000*(180/np.pi)**1
p0 = 973.3
S  = 0.44
d  = 0.75
m  = 13.98*32.174
ss = 1036.4
M  = 3
Iy = 182.5*32.174
g  = 32.2

def dynamicsf(x,p):
    """
    Input-affine nonlinear dynamical system of interest when u = 0
    (i.e. f of dx/dt = f(x)+g(x)u)

    Parameters
    ----------
    x : ndarray - (n, )
        current state
    p : ndarray - (n_p, )
        current system parameter (Mach number)

    Returns
    -------
    fx : ndarray - (n, )
        f(x) of dx/dt = f(x)+g(x)u

    """
    a = x[0]
    q = x[1]
    M = p[0]
    V  = M*ss
    u = V*np.cos(a)
    Cn = an*a**3+bn*np.abs(a)*a+cn*(2+M/3)*a
    Cm = am*a**3+bm*np.abs(a)*a-cm*(7-8*M/3)*a
    Fz = Cn*0.7*p0*M**2*S
    My = Cm*0.7*p0*M**2*S*d
    fa = np.cos(a)**2/m/u*Fz+q
    fq = My/Iy
    fx = np.array([fa,fq])
    return fx

def dynamicsg(x,p):
    """
    Nonlinear actuation matrix of input-affine nonlinear dynamical system 
    (i.e. g of dx/dt = f(x)+g(x)u)

    Parameters
    ----------
    x : ndarray - (n, )
        current state
    p : ndarray - (n_p, )
        current system parameter

    Returns
    -------
    gx : ndarray - (n,n)
        g(x) of dx/dt = f(x)+g(x)u

    """
    a = x[0]
    M = p[0]
    V  = M*ss
    u = V*np.cos(a)
    Cn = dn
    Cm = dm
    Fz = Cn*0.7*p0*M**2*S
    My = Cm*0.7*p0*M**2*S*d
    ga = np.cos(a)**2/m/u*Fz
    gq = My/Iy
    gx = np.array([[ga],[gq]])
    return gx

def getA(x,p):
    """
    State-dependent coefficient (SDC) matrix of given f(x,p) (i.e. A(x,p)x = f(x,p))

    Parameters
    ----------
    x : ndarray - (n, )
        current state
    p : ndarray - (n_p, )
        current system parameter

    Returns
    -------
    Ax : ndarray - (n, )
        A(x,p) of A(x,p)x = f(x,p)

    """
    a = x[0]
    M = p[0]
    V  = M*ss
    u = V*np.cos(a)
    A11 = np.cos(a)**2/m/u*0.7*p0*M**2*S*(an*a**2+bn*np.abs(a)+cn*(2+M/3))
    A21 = 0.7*p0*M**2*S*d/Iy*(am*a**2+bm*np.abs(a)-cm*(7-8*M/3))
    Ax = np.array([[A11,1],[A21,0]])
    return Ax

#ncmC = SNCM(dt,dynamicsf,dynamicsg,xlims,alims,elims,"con",fname,d1_over=1.5e-3,lam=1.5,da=0.1,plims=plims)
ncmC = NCM(dt,dynamicsf,dynamicsg,xlims,alims,"con",fname,d1_over=1.5e-3,d2_over=1.5,da=0.1,plims=plims)
ncmC.Afun = getA

# You can use ncm.train(iTrain = 0) instead when using pre-traind NCM models.
ncmC.train(iTrain = 1)
ncmC.model.trainable = False

# simultion time step
dt = 0.1
# terminal time
tf = 20
# initial state
x0 = np.random.uniform(low=xlims[0,:],high=xlims[1,:])
ncmC.Bw = dynamicsg
# simulation
snames = [r"$\alpha$",r"$q$"]
this,xhis,_ = ncmC.simulation(dt,tf,x0,dscale=1e2,xnames=snames,Ncol=2,FigSize=(20,10))