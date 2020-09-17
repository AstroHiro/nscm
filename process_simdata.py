#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:51:36 2020

@author: hiroyasu
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text', usetex=True)
FontSize = 15
LabelSize = 17
DPI = 300
atra = 0.8
nticks_acc = 25

fname1 = "models/optvals/NCMmissileE"
fname2 = "simdata"

alp_opt_s = np.load(fname1+"/alp_opt_s.npy")
eps_opt = np.load(fname1+"/eps_opt.npy")
nu_opt = np.load(fname1+"/nu_opt.npy")
chi_opt = np.load(fname1+"/chi_opt.npy")
Jcv_opt = np.load(fname1+"/Jcv_opt.npy")
ahis = np.load(fname1+"/ahis.npy")
ehis = np.load(fname1+"/ehis.npy")
Jhis = np.load(fname1+"/Jhis.npy")
c_opt = np.load(fname1+"/c_opt_est.npy")

this = np.load(fname2+"/this.npy")
xhis = np.load(fname2+"/xhis.npy")
zhis = np.load(fname2+"/zhis.npy")
x2his = np.load(fname2+"/x2his.npy")
z2his = np.load(fname2+"/z2his.npy")
x3his = np.load(fname2+"/x3his.npy")
z3his = np.load(fname2+"/z3his.npy")
x4his = np.load(fname2+"/x4his.npy")
z4his = np.load(fname2+"/z4his.npy")
x5his = np.load(fname2+"/x5his.npy")
z5his = np.load(fname2+"/z5his.npy")
uehis = np.load(fname2+"/uehis.npy")
ue2his = np.load(fname2+"/ue2his.npy")
ue3his = np.load(fname2+"/ue3his.npy")
ue4his = np.load(fname2+"/ue4his.npy")
ue5his = np.load(fname2+"/ue5his.npy")
d2Mcdx2his = np.load(fname2+"/d2Mcdx2his.npy")
d2Medx2his = np.load(fname2+"/d2Medx2his.npy")
Mc2his = np.load(fname2+"/Mc2his.npy")
Me2his = np.load(fname2+"/Me2his.npy")
Jcvc_opt = np.load(fname2+"/Jcvc_opt.npy")
Jcve_opt = np.load(fname2+"/Jcve_opt.npy")
Lmc = np.load(fname2+"/Lmc.npy")
Lme = np.load(fname2+"/Lme.npy")
nuc_opt = np.load(fname2+"/nuc_opt.npy")
nue_opt = np.load(fname2+"/nue_opt.npy")
chic_opt = np.load(fname2+"/chic_opt.npy")
chie_opt = np.load(fname2+"/chie_opt.npy")

Jhis[Jhis>=0.5] = 0.5

plt.figure()
X,Y = np.meshgrid(ahis,ehis)
cp = plt.contourf(X,Y,Jhis,20)
#.clabel(cp,colors="white",fmt="%2.1f",fontsize=13)
cb = plt.colorbar(cp)
cb.set_label("optimal estimation error",fontsize=LabelSize)
cb.set_ticks([0.378,0.396,0.414,0.432,0.450,0.468,0.486,0.500])
cb.set_ticklabels([r"0.378",r"0.396",r"0.414",r"0.432",r"0.450",r"0.468",r"0.486",r"$\geq 0.500$"])
plt.xlabel(r"contraction rate $\alpha$",fontsize=LabelSize)
plt.ylabel(r"design parameter $\varepsilon$",fontsize=LabelSize)
#plt.title(tit1,fontsize=FontSize)
fname = "figs/alpeps.pdf"
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()


data1 = {'score': np.sum((xhis-zhis)**2,1)}
data2 = {'score': np.sum((x2his-z2his)**2,1)}
data3 = {'score': np.sum((x3his-z3his)**2,1)}
data4 = {'score': np.sum((x4his-z4his)**2,1)}
data5 = {'score': np.sum((x5his-z5his)**2,1)}
# Create dataframe
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)
df5 = pd.DataFrame(data5)
WindowSize = 1
dfe1his = df1.rolling(window=WindowSize).mean()
dfe2his = df2.rolling(window=WindowSize).mean()
dfe3his = df3.rolling(window=WindowSize).mean()
dfe4his = df4.rolling(window=WindowSize).mean()
dfe5his = df5.rolling(window=WindowSize).mean()
plt.figure()
plt.plot(this,dfe2his,alpha=atra)
plt.plot(this,dfe5his,alpha=atra,color="#17becf")
plt.plot(this,dfe3his,alpha=atra,color="#2ca02c")
plt.plot(this,dfe4his,alpha=atra)
plt.plot(this,dfe1his,alpha=atra,color="#e377c2")
plt.plot(this,np.ones(np.size(this))*Jcve_opt,color="black",alpha=0.8)
plt.grid()
plt.xlabel(r"time $t$",fontsize=LabelSize)
plt.ylabel(r"estimation error $\|x(t)-\hat{x}(t)\|^2$",fontsize=LabelSize)
plt.legend([r"SDRE",r"EKF",r"NCM",r"mCV-STEM",r"NSCM",r"steady-state upper bound"],loc="upper right",bbox_to_anchor=(1,1.02))
plt.ylim([-0.05,3.4])
#plt.title(tit1,fontsize=FontSize)
fname = "figs/estimation.pdf"
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()


plt.figure()
plt.plot(this,np.sum((x2his)**2,1),alpha=atra)
plt.plot(this,np.sum((x5his)**2,1),alpha=atra,color="#17becf")
plt.plot(this,np.sum((x3his)**2,1),alpha=atra,color="#2ca02c")
plt.plot(this,np.sum((x4his)**2,1),alpha=atra)
plt.plot(this,np.sum((xhis)**2,1),alpha=atra,color="#e377c2")
plt.plot(this,np.ones(np.size(this))*Jcvc_opt,color="black",alpha=0.8)
plt.grid()
plt.xlabel(r"time $t$",fontsize=LabelSize)
plt.ylabel(r"tracking error $\|x(t)-x_d(t)\|^2$",fontsize=LabelSize)
plt.legend([r"SDRE",r"ILQR",r"NCM",r"mCV-STEM",r"NSCM",r"steady-state upper bound"],loc="upper right",bbox_to_anchor=(1,1.02))
#plt.title(tit1,fontsize=FontSize)
plt.ylim([-0.03,1.7])
fname = "figs/tracking.pdf"
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()

lratio = 1.8
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(14,4))
fig.subplots_adjust(wspace=0.3)
ax1.plot(this[0:-1],d2Mcdx2his/Lmc)
ax1.plot(this[0:-1],d2Medx2his/Lme)
ax1.plot(this[0:-1],np.ones(100),color="black",alpha=0.8)
#plt.title(tit1,fontsize=FontSize)
ax1.grid()
ax1.set_xlabel(r"time $t$",fontsize=LabelSize*lratio)
ax1.set_ylabel(r"$\max_{ij}\|X_{,x_ix_j}\|/L_m$",fontsize=LabelSize*lratio)
ax1.set_xticks([0,2,4,6,8,10])
ax1.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax1.tick_params(labelsize=nticks_acc)
ax1.set_ylim([-0.0166,1.033])
ax1.legend([r"NSCM control"],loc="upper right",bbox_to_anchor=(1,1.01),fontsize=23)

ax2.plot(this[0:-1],Me2his,color="#ff7f0e")
ax2.plot(this[0:-1],Mc2his,color="#1f77b4")
ax2.plot(this[0:-1],0.03*np.ones(100),color="black",alpha=0.8)
ax2.grid()
ax2.set_xlabel(r"time $t$",fontsize=LabelSize*lratio)
ax2.set_ylabel(r"prediction error",fontsize=LabelSize*lratio)
ax2.set_xticks([0,2,4,6,8,10])
ax2.set_yticks([0.00,0.01,0.02,0.03,0.04])
ax2.set_ylim([-0.0005,0.031])
ax2.tick_params(labelsize=nticks_acc)
ax2.legend([r"NSCM estimation"],loc="upper right",bbox_to_anchor=(1.01,1.01),fontsize=23)
fname = "figs/errors.pdf"
fig.savefig(fname,bbox_inches='tight',dpi=DPI)

fig = plt.figure()
ax = fig.add_subplot(111)

# Reversed Greys colourmap for filled contours
cpf = ax.contourf(X,Y,Jhis,20,cmap=cm.Greys_r)
# Set the colours of the contours and labels so they're white where the
# contour fill is dark (Z < 0) and black where it's light (Z >= 0)
colours = ['w' if level<0 else 'k' for level in cpf.levels]
cp = ax.contour(X,Y,Jhis,20,colors=colours)
ax.clabel(cp,fontsize=12,colors=colours)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Jhis,cmap=cm.coolwarm)



def bmotion(dt,x):
    dW = np.random.multivariate_normal(np.zeros(1),dt*np.identity(1))*0.01
    return x+dW
def bmotion2(dt,x):
    dW = np.random.multivariate_normal(np.zeros(1),dt*np.identity(1))*0.01
    return x+dW
dt = 1
t = 0
x = 0
x2 = 0
x3 = 0
N = 100
tbhis = np.zeros(N+1)
tbhis[0] = t
xbhis = np.zeros(N+1)
xbhis[0] = x
xbhis2 = np.zeros(N+1)
xbhis2[0] = x2
xbhis3 = np.zeros(N+1)
xbhis3[0] = x3
for i in range(N):
    t = t+dt
    x = bmotion(dt,x)
    x2 = bmotion(dt,x2)
    x3 = bmotion(dt,x3)
    tbhis[i+1] = t
    xbhis[i+1] = x
    xbhis2[i+1] = x2
    xbhis3[i+1] = x3


np.random.seed(886)
plt.figure(figsize=(10,2))
fname = "figs/bm1.png"
plt.plot(tbhis,xbhis,color="#005851",linewidth=3)
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()
plt.figure(figsize=(10,2))
fname = "figs/bm2.png"
plt.plot(tbhis,xbhis2,color="#005851",linewidth=3)
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()
plt.figure(figsize=(10,2))
fname = "figs/bm3.png"
plt.plot(tbhis,xbhis3,color="#005851",linewidth=3)
plt.savefig(fname,bbox_inches='tight',dpi=DPI)
plt.show()