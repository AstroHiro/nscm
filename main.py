#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:30:12 2020

@author: hiroyasu
"""

import numpy as np
from classsncm import SNCM
from classncm_sn import NCM
import tensorflow
import random as python_random

np.random.seed(7)
tensorflow.random.set_seed(226)
python_random.seed(226)
    
# enter your choice of CV-STEM sampling period
dt = 0.1
# specify upper and lower bounds of each state
xlims = np.array([-np.ones(2),np.ones(2)])*np.pi/5
#xlims = np.array([[-1,-2],[1,2]])*20*np.pi/180
plims = np.array([[2],[4]])
# specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
alims = np.array([0.1,30])
elims = np.array([0.1,10])
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

def dynamics(x,p):
    """
    Nonlinear dynamical system of interest
        

    Parameters
    ----------
    x : ndarray - (n, )
        current state
    p : ndarray - (n_p, )
        current system parameter

    Returns
    -------
    dxdt : ndarray - (n, )
        time derivative of x given by dynamical system of interest

    """
    return dynamicsf(x,p)

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

def measurement(x,u,p):
    """
    Nonlinear measurement equation
        

    Parameters
    ----------
    x : ndarray - (n, )
        current state
    u : ndarray - (m, )
        current input
    p : ndarray - (n_p, )
        current system parameter

    Returns
    -------
    y : ndarray - (n, )
        measurement, i.e. y of y = h(x) where h is measurement equation

    """
    a = x[0]
    q = x[1]
    M = p[0]
    Cn = an*a**3+bn*np.abs(a)*a+cn*(2+M/3)*a+dn*u[0]
    Fz = Cn*0.7*p0*M**2*S
    y = np.array([q,Fz/m/g])
    return y

def getC(x,p):
    """
    State-dependent coefficient (SDC) matrix of given h(x,u,p) when u = 0
        

    Parameters
    ----------
    x : ndarray - (n, )
        current state
    p : ndarray - (n_p, )
        current system parameter

    Returns
    -------
    Cx : ndarray - (n, )
        SCD matrix

    """
    a = x[0]
    M = p[0]
    Cx = np.array([[0,1],[0.7*p0*M**2*S*(an*a**2+bn*np.abs(a)+cn*(2+M/3))/m/g,0]])
    return Cx


# enter your choice of CV-STEM sampling period
dt = 0.1
# specify upper and lower bounds of each state
xlims = np.array([-np.ones(2),np.ones(2)])*np.pi/3
#xlims = np.array([[-1,-2],[1,2]])*20*np.pi/180
plims = np.array([[2],[4]])
# specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
alims = np.array([0.1,30])
elims = np.array([1,3])
# name your NCM
fname = "SNCMmissileC"

ncmC = SNCM(dt,dynamicsf,dynamicsg,xlims,alims,elims,"con",fname,d1_over=1.5e-3/2.5,lam=1.5/2.5,da=0.1,plims=plims,Lm=10,Lscale=3)
ncmCd = NCM(dt,dynamicsf,dynamicsg,xlims,alims,"con",fname,d1_over=1.5e-3/2.5,d2_over=1.5/2.5,da=0.1,plims=plims,Lm=10,Lscale=3)
ncmC.Afun = getA
ncmCd.Afun = getA

# You can use ncm.train(iTrain = 0) instead when using pre-traind NCM models.
ncmC.train()
ncmC.model.trainable = False
ncmCd.train(Verbose=0)
ncmCd.model.trainable = False

# simultion time step
dt = 0.1
# terminal time
tf = 20
# initial state
x0 = np.random.uniform(low=xlims[0,:],high=xlims[1,:])
ncmC.Bw = dynamicsg
# simulation
snames = [r"$\alpha$",r"$q$"]
this,xhis,_ = ncmC.simulation(dt,tf,x0,dscale=0.01,xnames=snames,Ncol=2,FigSize=(20,10))
#this,xhis,_ = ncmC.simulation(dt,tf,x0,dscale=1e2,xnames=snames,Ncol=2,FigSize=(20,10))


# enter your choice of CV-STEM sampling period
dt = 0.1
# specify upper and lower bounds of each state
xlims = np.array([-np.ones(2),np.ones(2)])*np.pi/3
# specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
alims = np.array([0.1,0.7])
elims = np.array([2.0,4.0])
# name your NCM
fname = "NCMmissileE"
meas_ncm = lambda x,p: measurement(x,np.zeros(1),p)

"""
ncmE = SNCM(dt,dynamics,meas_ncm,xlims,alims,elims,"est",fname,plims=plims,Lm=0.5,Lscale=4.5)
ncmEd = NCM(dt,dynamics,meas_ncm,xlims,alims,"est",fname,plims=plims,Lm=0.5,Lscale=4.5)
ncmE.Afun = lambda x,p: getA(x,p)
ncmE.Cfun = lambda x,p: getC(x,p)
ncmEd.Afun = lambda x,p: getA(x,p)
ncmEd.Cfun = lambda x,p: getC(x,p)
ncmE.linesearch_acc()
"""

alp_opt_s = np.load("models/optvals/"+fname+"/alp_opt_s.npy")
eps_opt = np.load("models/optvals/"+fname+"/eps_opt.npy")
#alims = np.array([alp_opt_s-0.1,alp_opt_s-0.1])
#elims = np.array([eps_opt-0.1,eps_opt-0.1])
alims = np.array([alp_opt_s,alp_opt_s])
elims = np.array([eps_opt,eps_opt])
alimsd = np.array([0.1,10])

ncmE = SNCM(dt,dynamics,meas_ncm,xlims,alims,elims,"est",fname,plims=plims,Lm=0.5,Lscale=4.5)
ncmEd = NCM(dt,dynamics,meas_ncm,xlims,alimsd,"est",fname,plims=plims,Lm=0.5,Lscale=4.5)
ncmE.Afun = lambda x,p: getA(x,p)
ncmE.Cfun = lambda x,p: getC(x,p)
ncmEd.Afun = lambda x,p: getA(x,p)
ncmEd.Cfun = lambda x,p: getC(x,p)
# You can use ncm.train(iTrain = 0) instead when using pre-traind NCM models.
ncmE.train()
ncmE.model.trainable = False
ncmEd.train(Verbose=0)
ncmEd.model.trainable = False


# simultion time step
dt = 0.1
# terminal time
tf = 10
# initial state
x0 = np.array([0.7,0.7])
# estimated initial state
z0 = np.random.uniform(low=xlims[0,:],high=xlims[1,:])
#z0 = np.array([-0.8,-0.8])
# simulation
snames = [r"$\alpha$",r"$q$"]
ncmE.Bw = dynamicsg
ncmE.Gw = lambda x,p: np.identity(2)
ncmEd.Bw = dynamicsg
ncmEd.Gw = lambda x,p: np.identity(2)
# weights for SDRE estimation and control
Qe = np.identity(2)*1.0
Qc = np.identity(2)*0.7
Re = np.identity(2)*0.1
Rc = np.identity(1)
this,xhis,zhis,x3his,z3his,ue3his,phis = ncmE.simulation_OFC(ncmE,ncmC,ncmEd,ncmCd,dynamicsf,dynamicsg,getC,measurement,dt,tf,x0,z0,Qe,Qc,Re,Rc,dscale=0.3,xnames=snames,Ncol=2,FigSize=(20,10),FontSize=20)
