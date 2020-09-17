#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Hiroyasu Tsukamoto https://hirotsukamoto.com/

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMI TED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

"""

import os
import cvxpy as cp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
from inspect import signature as sig
import control
from SpectralNormalizationKeras import DenseSN
from scipy.optimize import fsolve

class SNCM:
    def __init__(self,dt,dynamicsf,h_or_g,xlims,alims,elims,iEC,fname,d1_over=0.1,\
            lam=0.1,da=0.1,de=0.1,Nx=1000,Nls=100,plims=np.empty((2,0)),Lm=10,Lscale=1):
        """
        This class provides several objects and methods for designing a Neural 
        Contraction Metric (NCM) of a given nonliner dynamical system both for
        state estimation and feedback control.
        See the NCM paper https://arxiv.org/abs/2006.04361 and
        the CV-STEM paper https://arxiv.org/abs/2006.04359 for more details.
        See https://github.com/AstroHiro/ncm/wiki/Documentation for the
        documentation of this class file.
        
        
        Parameters
        (let n: state dimension and m: measurement or control input dimension)
        ----------
        dt : float
            discrete sampling period of CV-STEM
        dynamicsf : function - ndarray (n,n_p) -> (n, )
            vector field of given dynamical system 
            (i.e. f of dx/dt = f(x) or dx/dt = f(x)+g(x)u)
        h_or_g : function - ndarray (n,n_p) -> (m, ) for h, -> (n,m) for g
            measurement equation h or actuation matrix g
            (i.e. h of y = h(x,p) or g of dx/dt  = f(x,p)+g(x,p)u)
        xlims : ndarray - (2,n)
            lower and upper buonds of eash state
            (i.e. xlims[0,:]: lower bounds, xlims[1,:]: upper bounds)
        alims : ndarray - (2, )
            lower and upper bound of contraction rate alpha
            (i.e. alims[0]: lower bound, alims[0]: upper bound)
        iEC : str
            iEC = "est" for estimation and = "con" for control
        fname : str
            file name of your choice for storing NCM models and parameters
        d1_over : float, optional, default is 0.1
            upper bound of process noise
            (i.e. d1_over or d_over in the NCM paper)
        d2_over : float, optional, default is 0.1
            upper bound of measurement noise or penalty on feedback gains
            (i.e. d2_over or lambda in in the NCM paper)
        da : float, optional, default is 0.1
            step size of contraction rate alpha for line search in CV-STEM
        Nx : int, optional, default is 1000
            # samples of CV-STEM to be used for NCM training 
        Nls : int, optional, default is 100
            # samples to be used for line search in CV-STEM
        plims : ndarray - (2,n_p), default is np.empty((2,0))
            lower and upper bound of system parameters
            (i.e. plims[0,:]: lower bounds, plims[0,:]: upper bounds)

        Any other objects to be updated
        -------
        n : int
            state dimension
        m : int
            measurement or control input dimension
        n_p : int
            system parameter dimension
        Afun : function - ndarray (n,n_p) -> (n,n)
            Jacobian of dynamicsf (can be set to state-dependent coefficient
            matrix A s.t. f(x) = A(x)x, see the CV-STEM paper for details)
        Cfun : function - ndarray (n,n_p) -> (n,m), to be used for iEC = "est"
            Jacobian of measurement equation h (can be set to C s.t. 
            h(x) = C(x)x, see the CV-STEM paper for details)
        Bw : function - ndarray (n,n_p) -> (n,k1)
            B(x) given in equation (9) or B_2(x) in equation (17) of the NCM
            paper (B(x) = I and B_2(x) = g(x) are used by default, where g(x)
            is actuation matrix)
        Gw : function - ndarray (n,n_p) -> (m,k2), to be used for iEC = "est"
            G(x) given in equation (9) of the NCM paper (G(x) = I is used by
            default)
        c_over : numpy.float64, to be used for iEC = "est"
            approximate upper bound of Cfun(x) in given state space
        b_over : numpy.float64
            approximate upper bound of Bw(x) in given state space
        g_over : numpy.float64, to be used for iEC = "est"
            approximate upper bound of Gw(x) in given state space
        model : keras neural net model - ndarray (k,n) -> (k,int(n*(n+1)/2))
            function that returns cholesky-decomposed approximate optimal
            contraction metrics (i.e. NCMs) for given k states
        alp_opt : float
            optimal contraction rate
        chi_opt : numpy.float64
            optimal upper bound of condition number of contraction metrics
        nu_opt : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
        Jcv_opt : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
        xs_opt : ndarray - (Nx,n)
            randomized state samples
        Ws_opt : list of length Nx
            list containing inverse of ndarray (n,n) optimal contraction
            metrics sampled by CV-STEM
        Ms_opt : list of length Nx
            list containing ndarray (n,n) optimal contraction metrics sampled
            by CV-STEM
        cholMs : list of length Nx
            list containing ndarray (int(n*(n+1)/2), ) optimal contraction 
            metrics sampled by CV-STEM
        Ws : list of length Ncv
            list containing inverse of n by n optimal contraction metrics
            in current instance of CV-STEM
        chi : numpy.float64
            optimal upper bound of condition number of contraction metrics
            in current instance of CV-STEM
        nu : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
            in current instance of CV-STEM
        Jcv : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
            in current instance of CV-STEM
        cvx_status : str
            problem status of CV-STEM, "optimal", "infeasible", "unbounded",
            "infeasible_inaccurate", or "unbounded_inaccurate"
        dt_rk : float, default is 0.01
            time step of numerical integration 
            
        """
        self.dt = dt
        if (len(sig(dynamicsf).parameters) == 1):
            self.dynamicsf = lambda x,p: dynamicsf(x)
        else:
            self.dynamicsf = dynamicsf
        if (len(sig(h_or_g).parameters) == 1):
            self.h_or_g =  lambda x,p: h_or_g(x)
        else:
            self.h_or_g = h_or_g
        self.xlims = xlims
        self.alims = alims
        self.elims = elims
        self.iEC = iEC
        self.fname = fname
        self.d1_over = d1_over
        self.lam = lam
        self.da = da
        self.de = de
        self.Nx = Nx
        self.Nls = Nls
        self.n = np.size(xlims,1)
        self.m = np.size(self.h_or_g(xlims[0,:],plims[0,:]).T,0)
        self.n_p = np.size(plims,1)
        self.Afun= lambda x,p: self.jacobian(x,p,self.dynamicsf)
        if self.iEC == "est":
            self.Cfun= lambda x,p: self.jacobian(x,p,self.h_or_g)
            self.Bw = lambda x,p: np.identity(self.n)*1e-2*np.sqrt(10)
            self.Gw = lambda x,p: np.identity(self.m)*1e-2*np.sqrt(10)
        elif self.iEC == "con":
            #self.Bw = self.h_or_g
            self.Bw = lambda x,p: np.identity(self.n)*0.1/np.sqrt(2.5)
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        self.dt_rk = 0.01
        self.plims = plims
        self.Lm = Lm
        self.Lscale = Lscale
    
    def ncm(self,x,p):
        """
        Compute Neural Contraction Metric (NCM) M(x,p) at current state x and 
        current system parameter p
        
        
        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter

        Returns
        -------
        M : ndarray - (n,n)
            Neural Contraction Metric (NCM)
            
        """
        n = self.n
        n_p = self.n_p
        x = np.reshape(np.hstack((x,p)),(1,n+n_p))
        cholM = self.model.predict(x)
        cholM = np.reshape(cholM,int(n*(n+1)/2))
        M = self.cholM2M(cholM)
        return M
    
    def train(self,iTrain=1,Nbatch=32,Nlayers=3,Nunits=100,Nepochs=10000,
              ValidationSplit=0.1,Patience=20,Verbose=2):
        """
        Train Neural Contraction Metric (NCM)
        

        Parameters
        ----------
        iTrain : 1 or 0, optional, default is 1
            IdXTrain = 1 for training NCM and = 0 for using pretrained NCM
        Nbatch : int, optional, default is 32
            batch size of NCM training
        Nlayers : int, optional, default is 3
            # layers of NCM 
        Nunits : int, optional, default is 100
            # units of each layers of NCM
        Nepochs : int, optional, default is 10000
            # training epochs
        ValidationSplit : int, optional, default is 0.1
            proportion of training data used as verification data
        Patience : int, optional, default is 20
            # epochs with no improvement after which training will be stopped
        Verbose : 0, 1, or 2, optional, default is 2
            # verbosity mode (0: silent 1: progress bar 2: one line per epoch)

        Objects to be updated
        -------
        model : keras neural net model - ndarray (k,n) -> (k,int(n*(n+1)/2))
            function that returns cholesky-decomposed approximate optimal
            contraction metrics (i.e. NCMs) for given k states
            
        When iTrain = 0, follwoing objects will also be updated
        alp_opt : float
            optimal contraction rate
        chi_opt : numpy.float64
            optimal upper bound of condition number of contraction metrics
        nu_opt : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
        Jcv_opt : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
            
        """
        if iTrain == 1:
            self.cvstem()
            print("========================================================")
            print("=================== NCM CONSTRUCTION ===================")
            print("========================================================")
            n = self.n
            nout = int(n*(n+1)/2)
            c_opt = self.spectral_norm_const(Nlayers,Nunits)
            X = np.hstack((self.xs_opt,self.ps_opt))
            y = self.cholMs
            model = Sequential(name="NCM")
            model.add(DenseSN(Nunits,kernel_initializer="glorot_uniform"))
            model.add(Lambda(lambda x: x*c_opt))
            model.add(Activation("tanh"))
            for l in range(Nlayers-1):
                model.add(DenseSN(Nunits,kernel_initializer="glorot_uniform"))
                model.add(Lambda(lambda x: x*c_opt))
                model.add(Activation("tanh"))
            model.add(DenseSN(nout,kernel_initializer='glorot_uniform'))
            model.add(Lambda(lambda x: x*np.sqrt(self.mbar)))
            model.compile(loss="mean_squared_error",optimizer="adam")
            es = EarlyStopping(monitor="val_loss",patience=Patience)
            model.fit(X,y,batch_size=Nbatch,epochs=Nepochs,verbose=Verbose,\
                      callbacks=[es],validation_split=ValidationSplit)
            self.model = model
            #model.save("models/"+self.fname+".h5")
        elif iTrain == 0:
            print("========================================================")
            print("=================== NCM CONSTRUCTION ===================")
            print("========================================================")
            self.model = load_model("models/"+self.fname+".h5")
            path = "models/optvals/"+self.fname
            self.alp_opt = np.load(path+"/alp_opt.npy")
            self.chi_opt = np.load(path+"/chi_opt.npy")
            self.nu_opt = np.load(path+"/nu_opt.npy")
            self.Jcv_opt = np.load(path+"/Jcv_opt.npy")
            print("Loading pre-trained NCM ...")
            print("Loading pre-trained NCM END")
        else:
            raise ValueError("Invalid iTrain: iTrain = 1 or 0")
        print("========================================================")
        print("================= NCM CONSTRUCTION END =================")
        print("========================================================")
        pass
      
    def cvstem(self):
        """        
        Sample optimal contraction metrics by CV-STEM for constructing NCM


        Objects to be updated
        -------
        c_over : numpy.float64, to be used for iEC = "est"
            Approximate upper bound of Cfun(x) in given state space
        b_over : numpy.float64
            Approximate upper bound of Bw(x) in given state space
        g_over : numpy.float64, to be used for iEC = "est"
            Approximate upper bound of Gw(x) in given state space
        xs_opt : ndarray - (Nx,n), where Nx is # samples to be used for NCM
            randomized state samples
        Ws_opt : list of length Nx
            list containing inverse of ndarray (n,n) optimal contraction
            metrics
        chi_opt : numpy.float64
            optimal upper bound of condition number of contraction metrics
        nu_opt : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
        Jcv_opt : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
        
        """
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun1 = self.Cfun
            self.Cfun = lambda x,p: fun1(x)
        if (self.iEC == "est") and (len(sig(self.Gw).parameters) == 1):
            fun2 = self.Gw
            self.Gw = lambda x,p: fun2(x)
        if self.iEC == "est":
            self.c_over = self.matrix_2bound(self.Cfun)
            self.g_over = self.matrix_2bound(self.Gw)
        if (len(sig(self.Bw).parameters) == 1):
            fun3 = self.Bw
            self.Bw = lambda x,p: fun3(x)
        self.b_over = self.matrix_2bound(self.Bw)
        self.linesearch()
        alp = self.alp_opt
        eps = self.eps_opt
        Nx = self.Nx
        Nsplit = 1
        Np = int(Nx/Nsplit)
        Nr = np.remainder(Nx,Nsplit)
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(Nx,Nxp))
        xs_opt,ps_opt,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        Ws_opt = []
        chi_opt = 0
        nu_opt = 0
        print("========================================================")
        print("====== SAMPLING OF CONTRACTION METRICS BY CV-STEM ======")
        print("========================================================")
        for p in range(Np):
            if np.remainder(p,int(Np/10)) == 0:
                print("# sampled metrics: ",p*Nsplit,"...")
            xs_p = xs_opt[Nsplit*p:Nsplit*(p+1),:]
            ps_p = ps_opt[Nsplit*p:Nsplit*(p+1),:]
            self.cvstem0(xs_p,ps_p,alp,eps)
            Ws_opt += self.Ws
            if self.nu >= nu_opt:
                nu_opt = self.nu
            if self.chi >= chi_opt:
                chi_opt = self.chi
        if Nr != 0:
            print("# samples metrics: ",Nx,"...")
            xs_p = xs_opt[Nsplit*(p+1):Nx,:]
            ps_p = ps_opt[Nsplit*(p+1):Nx,:]
            self.cvstem0(xs_p,ps_p,alp,eps)
            Ws_opt += self.Ws
            if self.nu >= nu_opt:
                nu_opt = self.nu
            if self.chi >= chi_opt:
                chi_opt = self.chi
        self.xs_opt = xs_opt
        self.ps_opt = ps_opt
        self.Ws_opt = Ws_opt
        self.chi_opt = chi_opt
        self.nu_opt = nu_opt
        if self.iEC == "est":
            Ce1 = self.b_over**2*(2/eps+1)
            Ce2 = self.c_over**2*self.g_over**2*(2/eps+1)
            self.Jcv_opt = (Ce1*chi_opt+Ce2*chi_opt*nu_opt**2)/2/alp
            print("Optimal steady-state estimation error =",\
                  "{:.2f}".format(self.Jcv_opt))
        elif self.iEC == "con":
            self.Jcv_opt = self.b_over**2*(2/eps+1)*chi_opt/2/alp
            print("Optimal steady-state tracking error =",\
                  "{:.2f}".format(self.Jcv_opt))
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        self.M2cholM()
        path = "models/optvals/"+self.fname
        if os.path.exists(path) == False:
            try:
                os.makedirs(path)
            except: 
                raise OSError("Creation of directory %s failed" %path)
            else:
                print ("Successfully created directory %s " %path)
        else:
            print ("Directory %s already exists" %path)
        np.save(path+"/alp_opt.npy",alp)
        np.save(path+"/chi_opt.npy",self.chi_opt)
        np.save(path+"/nu_opt.npy",self.nu_opt)
        np.save(path+"/Jcv_opt.npy",self.Jcv_opt)
        print("========================================================")
        print("==== SAMPLING OF CONTRACTION METRICS BY CV-STEM END ====")
        print("========================================================\n\n")
        pass
    
    def linesearch(self):
        """
        Perform line search of optimal contraction rate in CV-STEM


        Objects to be updated
        -------
        alp_opt : float
            optimal contraction rate

        """
        alp = self.alims[0]
        da = self.da
        Na = int((self.alims[1]-self.alims[0])/da)+1
        eps = self.elims[0]
        de = self.de
        Ne = int((self.elims[1]-self.elims[0])/de)+1
        Jcv_prev = np.Inf
        Ncv = self.Nls
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(Ncv,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        print("========================================================")
        print("=========== LINE SEARCH OF ALPHA AND EPSILON ===========")
        print("========================================================")
        alp_opts = []
        for ke in range(Ne):
            for ka in range(Na):
                self.cvstem0(xs,ps,alp,eps)
                print("Optimal value: Jcv =","{:.2f}".format(self.Jcv),\
                      "( alpha =","{:.3f}".format(alp),")")
                if Jcv_prev <= self.Jcv:
                    alp = alp-da
                    break
                alp += da
                Jcv_prev = self.Jcv
            Jcv_prev = np.Inf
            alp_opts.append(alp)
            alp = self.alims[0]
            eps += de
        self.alp_opt = min(alp_opts)
        self.eps_opt = self.elims[0]+de*np.argmin(alp_opts)
        alp_opt = self.alp_opt
        print("Optimal contraction rate: alpha =","{:.3f}".format(alp_opt))
        print("Optimal epsilon: epsilon =","{:.3f}".format(self.eps_opt))
        print("========================================================")
        print("========= LINE SEARCH OF ALPHA AND EPSILON END =========")
        print("========================================================\n\n")
        pass
    
    def linesearch_acc(self):
        """
        Perform line search of optimal contraction rate in CV-STEM


        Objects to be updated
        -------
        alp_opt : float
            optimal contraction rate

        """
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun1 = self.Cfun
            self.Cfun = lambda x,p: fun1(x)
        if (self.iEC == "est") and (len(sig(self.Gw).parameters) == 1):
            fun2 = self.Gw
            self.Gw = lambda x,p: fun2(x)
        if self.iEC == "est":
            self.c_over = self.matrix_2bound(self.Cfun)
            self.g_over = self.matrix_2bound(self.Gw)
        if (len(sig(self.Bw).parameters) == 1):
            fun3 = self.Bw
            self.Bw = lambda x,p: fun3(x)
        self.b_over = self.matrix_2bound(self.Bw)
        alp = self.alims[0]
        da = self.da
        Na = int((self.alims[1]-self.alims[0])/da)+1
        eps = self.elims[0]
        de = self.de
        Ne = int((self.elims[1]-self.elims[0])/de)+1
        Ncv = self.Nls
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(Ncv,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        print("========================================================")
        print("=========== LINE SEARCH OF ALPHA AND EPSILON ===========")
        print("========================================================")
        alp_opts = []
        ahis = []
        ehis = []
        Jhis = np.zeros((Ne,Na))
        path = "models/optvals/"+self.fname
        for ke in range(Ne):
            ehis.append(eps)
            for ka in range(Na):
                if ke == 0:
                    ahis.append(alp)
                self.cvstem0(xs,ps,alp,eps)
                print("Optimal value: Jcv =","{:.2f}".format(self.Jcv),\
                      "( alpha =","{:.3f}".format(alp),")","( epsilon =","{:.3f}".format(eps),")")
                if self.iEC == "est":
                    Ce1 = self.b_over**2*(2/eps+1)
                    Ce2 = self.c_over**2*self.g_over**2*(2/eps+1)
                    Jcv = (Ce1*self.chi+Ce2*self.chi*self.nu**2)/2/alp
                elif self.iEC == "con":
                    Jcv = self.b_over**2*(2/eps+1)*self.chi/2/alp
                else:
                    raise ValueError('Invalid iEC: iEC = "est" or "con"')
                Jhis[ke,ka] = Jcv
                alp += da
                np.save(path+"/ahis.npy",ahis)
                np.save(path+"/ehis.npy",ehis)
                np.save(path+"/Jhis.npy",Jhis)
            alp_opts.append(ahis[np.argmin(Jhis[ke,:])])
            alp = self.alims[0]
            eps += de
        self.alp_opt = min(alp_opts)
        self.eps_opt = self.elims[0]+de*np.argmin(alp_opts)
        alp_opt = self.alp_opt
        self.ahis = ahis
        self.ehis = ehis
        self.Jhis = Jhis
        print("Optimal contraction rate: alpha =","{:.3f}".format(alp_opt))
        print("Optimal epsilon: epsilon =","{:.3f}".format(self.eps_opt))
        print("========================================================")
        print("========= LINE SEARCH OF ALPHA AND EPSILON END =========")
        print("========================================================\n\n")
        plt.figure()
        for ke in range(Ne):
            eps = ehis[ke]
            for ka in range(Na):
                plt.plot(ahis,Jhis[ke,:])
        np.save(path+"/alp_opt_s.npy",self.alp_opt)
        np.save(path+"/eps_opt.npy",self.eps_opt)
        np.save(path+"/ahis.npy",self.ahis)
        np.save(path+"/ehis.npy",self.ehis)
        np.save(path+"/Jhis.npy",self.Jhis)
        pass
    
    def cvstem0(self,xs,ps,alp,eps):
        """
        Run one single instance of CV-STEM algorithm for given states xs and
        contraction rate alpha


        Parameters
        ----------
        xs : ndarray - (Ncv,n), where Ncv is # state samples
            state samples for solving CV-STEM
        ps : ndarray - (Ncv,n_p), where Ncv is # state samples
            system parameter samples for solving CV-STEM
        alp : float
            contraction rate of interest

        Objects to be updated
        -------
        Ws : list of length Ncv
            list containing inverse of n by n optimal contraction metrics
            in current instance of CV-STEM
        chi : numpy.float64
            optimal upper bound of condition number of contraction metrics
            in current instance of CV-STEM
        nu : numpy.float64
            optimal upper bound of induced 2-norm of contraction metrics
            in current instance of CV-STEM
        Jcv : numpy.float64
            optimal steady-state upper bound of estimation or tracking error
            in current instance of CV-STEM
        cvx_status : str
            problem status of CV-STEM, "optimal", "infeasible", "unbounded",
            "infeasible_inaccurate", or "unbounded_inaccurate"

        """
        Ncv = np.size(xs,0)
        n = self.n
        Lm = self.Lm
        I = np.identity(n)
        Ws = []
        for k in range(Ncv):
            Ws.append(cp.Variable((n,n),PSD=True))
        nu = cp.Variable(nonneg=True)
        nu_c = cp.Variable(nonneg=True)
        chi = cp.Variable(nonneg=True)
        errtxt = "https://github.com/AstroHiro/ncm#troubleshooting"
        if len(sig(self.Afun).parameters) == 1:
            fun1 = self.Afun
            self.Afun = lambda x,p: fun1(x)
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun2 = self.Cfun
            self.Cfun = lambda x,p: fun2(x)
        if self.iEC == "est":
            Af = self.Afun
            Cf = self.Cfun
            a_e1 = Lm*self.b_over**2*(eps+1/2)
            a_e2 = Lm*self.c_over**2*self.g_over**2*(eps+1/2)
            J = (np.sqrt(3*self.b_over**2*(2/eps+1))*chi\
                 +np.sqrt(self.c_over**2*self.g_over**2*(2/eps+1))*nu)/alp**(1/3)
        elif self.iEC == "con":
            Af = lambda x,p: self.Afun(x,p).T
            Cf = lambda x,p: self.h_or_g(x,p).T
            a_gc = Lm*self.b_over**2*(eps+1/2)
            J = self.b_over**2*(2/eps+1)*chi/2/alp+self.lam*nu
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        constraints = []
        for k in range(Ncv):
            x = xs[k,:]
            p = ps[k,:]
            Ax = Af(x,p)
            Cx = Cf(x,p)
            W = Ws[k]
            constraints += [chi*I-W >> 0,W-I >> 0]
            con = -2*alp*W-((W-I)/self.dt+W@Ax+Ax.T@W-2*nu*Cx.T@Cx)
            if self.iEC == "est":
                M3 = con-a_e1*I-nu_c*a_e2*I
                constraints += [M3 >> 0]
            if self.iEC == "con":
                M1 = cp.hstack((con,W))
                M2 = cp.hstack((W,nu*I/a_gc))
                M12 = cp.vstack((M1,M2))
                constraints += [M12 >> 0]
        constraints += [cp.power(nu,3) <= nu_c]
        prob = cp.Problem(cp.Minimize(J),constraints)
        prob.solve(solver=cp.MOSEK)
        cvx_status = prob.status
        if cvx_status in ["infeasible","infeasible_inaccurate"]:
            raise ValueError("Problem infeasible: see "+errtxt)
        elif cvx_status in ["unbounded","unbounded_inaccurate"]:
            raise ValueError("Problem unbounded: "+errtxt)
        Wsout = []
        for k in range(Ncv):
            Wk = Ws[k].value/nu.value
            Wsout.append(Wk)
        self.Ws = Wsout
        self.nu = nu.value
        self.chi = chi.value
        self.Jcv = prob.value
        self.cvx_status = cvx_status
        pass
    
    def M2cholM(self):
        """
        Compute cholesky-decomposed optimal contraction metrics obtained by
        CV-STEM
        
        
        Objects to be updated
        -------
        Ms_opt : list of length Nx, where Nx is # samples to be used for NCM
            list containing ndarray (n,n) optimal contraction metrics
        cholMs : list of length Nx
            list containing ndarray (int(n*(n+1)/2), ) optimal contraction 
            metrics

        """
        Nx = self.Nx
        n = self.n
        Ms_opt = []
        cholMs = []
        for k in range(Nx):
            Mk = np.linalg.inv(self.Ws_opt[k])
            cholMk = np.linalg.cholesky(Mk)
            cholMk = cholMk.T # upper triangular
            cholMk_vec = np.zeros(int(n*(n+1)/2)) 
            for i in range (n):
                j = (n-1)-i;
                di = np.diag(cholMk,j)
                cholMk_vec[int(1/2*i*(i+1)):int(1/2*(i+1)*(i+2))] = di
            Ms_opt.append(Mk)
            cholMs.append(cholMk_vec)
        self.Ms_opt = Ms_opt
        self.cholMs = np.array(cholMs)
        pass
    
    def cholM2M(self,cholM):
        """
        Convert cholesky-decomposed optimal contraction metrics to original
        form in R^(n x n)


        Parameters
        ----------
        cholM : ndarray - (int(n*(n+1)/2), )
            cholesky-decomposed optimal contraction metrics

        Returns
        -------
        M : ndarray - (n,n)
            optimal contraction metrics

        """
        cMnp = 0
        n = self.n
        for i in range(n):
            lb = int(i*(i+1)/2)
            ub = int((i+1)*(i+2)/2)
            Di = cholM[lb:ub]
            Di = np.diag(Di,n-(i+1))
            cMnp += Di
        M = (cMnp.T)@cMnp
        return M  

    def d2Mdx2bound(self):
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(self.Nx,Nxp))
        xps2 = np.random.uniform(xpmin,xpmax,size=(self.Nx,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        xs2,ps2,_ = np.hsplit(xps2,np.array([self.n,Nxp]))
        n = self.n
        mat_over_out = 0
        dx = 0.01
        for k in range(self.Nx):
            xk = xs[k,:]
            pk = ps[k,:]
            xk2 = xs2[k,:]
            for i in range(n):
                ei = np.zeros(n)
                ei[i] = 1
                dxi = ei*dx
                dMi = (self.ncm(xk+dxi,pk)-self.ncm(xk,pk))/dx
                dMi2 = (self.ncm(xk2+dxi,pk)-self.ncm(xk2,pk))/dx
                mat_over = np.linalg.norm(dMi-dMi2,ord=2)
                mat_over = mat_over/np.linalg.norm(xk-xk2)
                print(mat_over)
                if mat_over > mat_over_out:
                    mat_over_out = mat_over
        return mat_over_out

    def dMdxbound(self):
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(self.Nx,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        n = self.n
        mat_over_out = 0
        dx = 0.01
        for k in range(self.Nx):
            xk = xs[k,:]
            pk = ps[k,:]
            for i in range(n):
                ei = np.zeros(n)
                ei[i] = 1
                dxi = ei*dx
                dMi = (self.ncm(xk+dxi,pk)-self.ncm(xk,pk))/dx
                mat_over = np.linalg.norm(dMi,ord=2)
                print(mat_over)
                if mat_over > mat_over_out:
                    mat_over_out = mat_over
        return mat_over_out

    def d2Mdx2bound2(self):
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(self.Nx,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        n = self.n
        mat_over_out = 0
        dx = 0.01
        for k in range(self.Nx):
            xk = xs[k,:]
            pk = ps[k,:]
            for i in range(n):
                ei = np.zeros(n)
                ei[i] = 1
                dxi = ei*dx
                for j in range(self.n):
                    ej = np.zeros(n)
                    ej[j] = 1
                    dxj = ej*dx
                    Mdij = self.ncm(xk+dxi+dxj,pk)
                    Mdi = self.ncm(xk+dxi,pk)
                    Mdj = self.ncm(xk+dxj,pk)
                    M = self.ncm(xk,pk)
                    dMij = (Mdij-Mdi-Mdj+M)/dx**2
                    mat_over = np.linalg.norm(dMij,ord=2)
                    print(mat_over)
                    if mat_over > mat_over_out:
                        mat_over_out = mat_over
        return mat_over_out

    def spectral_norm_const(self,Nlayers,Nunits):
        L = Nlayers
        U = Nunits
        Lm = self.Lm
        Lscale = self.Lscale
        if self.iEC == "est":
            self.mbar = self.nu_opt
        elif self.iEC == "con":
            self.mbar = self.chi_opt/self.nu_opt
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        th_ub = np.sqrt(self.mbar)
        thx_ub = lambda c: np.sqrt(self.mbar)*c**L
        thxx_ub = lambda c: np.sqrt(self.mbar)*(c**L-1)/(c-1)*c**(L+1)
        fun = lambda c: 2*(thx_ub(c)*thx_ub(c)+th_ub*thxx_ub(c)-Lm*Lscale)
        c_opt = fsolve(fun,1.1)   
        path = "models/optvals/"+self.fname
        if self.iEC == "est":
            np.save(path+"/c_opt_est.npy",c_opt)
        elif self.iEC == "con":
            np.save(path+"/c_opt_con.npy",c_opt)
        return c_opt
    
    def jacobian(self,x,p,fun):
        """
        Compute Jacobian of given vector field


        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        fun : function - ndarray (n, ) -> (nout, )
            given vector field

        Returns
        -------
        dfdx : ndarray - (ny,n)
             Jacobian of given vector field

        """
        n = self.n
        y = fun(x,p)
        h = 1e-4
        nout = np.size(y)
        dfdx = np.zeros((nout,n))
        for j in range(n):
            dx1 = np.zeros(n)
            dx2 = np.zeros(n)
            dx1[j] = -h
            dx2[j] = h
            dfdx[:,j] = (fun(x+dx2,p)-fun(x+dx1,p))/(2*h)
        return dfdx
    
    def matrix_2bound(self,fun):
        """
        Compute approximate upper bound of induced 2-norm of given matrix 
        function in given state space


        Parameters
        ----------
        fun : function - ndarray (n, ) -> (n1,n2)
            given matrix function

        Returns
        -------
        mat_over_out : numpy.float64
            upper bound of induced 2-norm of given matrix function in given 
            state space

        """
        xpmin = np.hstack((self.xlims[0,:],self.plims[0,:]))
        xpmax = np.hstack((self.xlims[1,:],self.plims[1,:]))
        Nxp = self.n+self.n_p
        xps = np.random.uniform(xpmin,xpmax,size=(self.Nx,Nxp))
        xs,ps,_ = np.hsplit(xps,np.array([self.n,Nxp]))
        mat_over_out = 0
        for k in range(self.Nx):
            Mat = fun(xs[k,:],ps[k,:])
            mat_over = np.linalg.norm(Mat,ord=2)
            if mat_over > mat_over_out:
                mat_over_out = mat_over
        return mat_over_out
    
    def dynamics(self,x,p,dEf):
        """
        Compute vector field of given nonlinear dynamical system with state 
        feedback input
        

        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter
        dEf : function - ndarray (n, ) -> (n, )
            function that returns state feedback input at current state

        Returns
        -------
        fout : ndarray - (n, )
            vector field of given nonliner dynamical system with state
            feedback input

        """
        fout = self.dynamicsf(x,p)+dEf(x,p)
        return fout

    def rk4(self,x,p,dEf,fun):
        """
        Compute state at next time step by 4th order Runge-Kutta method
        

        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter
        dEf : function - ndarray (n, ) -> (n, )
            function that returns state feedback input at current state
        fun : function - ndarray (n, ) -> (n, )
            function to be integrated

        Returns
        -------
        x : ndarray - (n, )
            state at next time step

        """
        Nrk = self.Nrk
        dt_rk = self.dt_rk
        for num in range(0,Nrk):
            k1 = fun(x,p,dEf)
            k2 = fun(x+k1*dt_rk/2.,p,dEf)
            k3 = fun(x+k2*dt_rk/2.,p,dEf)
            k4 = fun(x+k3*dt_rk,p,dEf)
            x = x+dt_rk*(k1+2.*k2+2.*k3+k4)/6.
        return x
    
    def unifrand2(self,d_over,nk):
        """
        Generate nk-dimensional random point uniformally distributed in
        L2-ball of radius d_over
        

        Parameters
        ----------
        d_over : float
            radius of L2-ball
        nk : int
            dimension of output vector

        Returns
        -------
        d : ndarray - (nk, )
            nk-dimensional random point uniformally distributed in L2-ball of 
            radius d_over

        """
        d_over_out = d_over+1
        while d_over_out > d_over:
            d = np.random.uniform(-d_over,d_over,size=nk)
            d_over_out = np.linalg.norm(d)
        return d

    
    def clfqp(self,x,p):
        """
        Compute optimal control input solving Control Layapunov Fucntion
        Quadratic Program (CLFQP) using NCM as Lyapunov function
        

        Parameters
        ----------
        x : ndarray - (n, )
            current state x
        p : ndarray - (n_p, )
            current system parameter

        Returns
        -------
        u : ndarray - (m, )
            current input u

        """
        alp = self.alp_opt
        nu = self.nu_opt
        dt = self.dt
        n = self.n
        I = np.identity(n)
        M = self.ncm(x,p)
        nu = np.size(self.h_or_g(x,p),1)
        u = cp.Variable((nu,1))
        e = np.reshape(x,(n,1))
        fx = np.reshape(self.dynamicsf(x,p),(n,1))
        gx = self.h_or_g(x,p)
        dMdt = (nu*I-M)/dt
        constraints = [2*e.T@(fx+gx@u)+e.T@dMdt@e <= -2*alp*e.T@M@e]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(u)),constraints)
        prob.solve()
        u = u.value
        u = np.ravel(u)
        return u
    
    def simulation(self,dt,tf,x0,z0=None,dscale=10.0,xnames="num",Ncol=1,\
                   FigSize=(20,10),FontSize=20,phis=None):
        """
        Perform NCM-based estimation or control of given nolinear dynamical
        systems and return simulation results
        

        Parameters
        ----------
        dt : float
            simulation time step
        tf : float
            terminal time
        x0 : ndarray - (n, )
            initial state
        z0 : ndarray - (n, ), to be used for iEC = "est"
            estimated initial state
        dscale : float, optional, default is 10
            scale of external disturbance 
        xnames : str, optional, default is "num"
            list containing names of each state, when xnames = "num", they are
            denoted as xnames = ["state 1","state 2",...]
        Ncol : int, optional, default is 1
            # columns of state figures to be generated
        FigSize : tuple, optional, default is (20,10)
            size of state figures to be generated
        FontSize : float, optional, default is 20
            font size of figures to be generated
        phis : system parameter history, optional, default is None
            history of system parameters

        Returns
        -------
        this : ndarray - (int(tf/dt)+1, )
            time histry 
        xhis : ndarray - (int(tf/dt)+1,n)
            state history
        zhis : ndarray - (int(tf/dt)+1,n), to be used for estimation tasks
            estimated state history

        """
        """
        
        
        1) SIMULATION
    
        
        """
        print("========================================================")
        print("====================== SIMULATIOM ======================")
        print("========================================================")
        if dt <= self.dt_rk:
            self.dt_rk = dt
        self.Nrk = int(dt/self.dt_rk)
        Nsim = int(tf/dt)
        np.set_printoptions(precision=1)
        print("time step =",dt)
        print("terminal time =",tf)
        print("initial state =",x0)
        k1 = np.size(self.Bw(self.xlims[0,:],self.plims[0,:]),1)
        Ik1 = np.identity(k1)
        if self.iEC == "est":
            print("estimated initial state =",z0)
            funx = lambda x,p,u: self.dynamicsf(x,p)
            funz = self.dynamics
            z = z0
            zhis = np.zeros((Nsim+1,self.n))
            zhis[0,:] = z
            tit1 = "Performance of NCM-based Estimation (1)"
            tit2 = "Performance of NCM-based Estimation (2)"
            ly = r"estimation error: $\|x-\hat{x}\|^2$"
            l1 = r"estimation error"
            k2 = np.size(self.Gw(self.xlims[0,:],self.plims[0,:]),1)
            Ik2 = np.identity(k2)
            bNam1 = "=================== ESTIMATION ERROR ==================="
            bNam2 = "============ ESTIMATION ERROR OF EACH STATE ============"
        elif self.iEC == "con":
            funx = self.dynamics
            zhis = np.zeros((Nsim+1,self.n))
            tit1 = "Performance of NCM-based Control (1)"
            tit2 = "Performance of NCM-based Control (2)"
            ly = r"tracking error: $\|x-x_d\|^2$"
            l1 = r"tracking error"
            bNam1 = "==================== TRACKING ERROR ===================="
            bNam2 = "============= TRACKING ERROR OF EACH STATE ============="
        else:
            raise ValueError('Invalid iEC: iEC = "est" or "con"')
        l2 = r"optimal steady-state upper bound"
        x = x0
        xhis = np.zeros((Nsim+1,self.n))
        xhis[0,:] = x
        this = np.linspace(0,tf,Nsim+1)
        if phis == None:
            phis = np.linspace(self.plims[0,:],self.plims[1,:],Nsim)
        if (self.iEC == "est") and (len(sig(self.Cfun).parameters) == 1):
            fun1 = self.Cfun
            self.Cfun = lambda x,p: fun1(x)
        if (len(sig(self.Bw).parameters) == 1):
            fun2 = self.Bw
            self.Bw = lambda x,p: fun2(x)
        if (self.iEC == "est") and (len(sig(self.Gw).parameters) == 1):
            fun3 = self.Gw
            self.Gw = lambda x,p: fun3(x)
        for k in range(Nsim):
            p = phis[k,:]
            if self.iEC == "est":
                Mx = self.ncm(z,p)
                Cx = self.Cfun(z,p)
                Lx = Mx@Cx.T
                dW2 = np.random.multivariate_normal(np.zeros(k2),dt*Ik2)*dscale
                y = self.h_or_g(x,p)+self.Gw(x,p)@dW2/dt
                dEf = lambda z,p: Lx@(y-self.h_or_g(z,p))
                z = self.rk4(z,p,dEf,funz)
                zhis[k+1,:] = z
            elif self.iEC == "con":
                Mx = self.ncm(x,p)
                Bx = self.h_or_g(x,p)
                Kx = Bx.T@Mx
                u = -Kx@x
                dEf = lambda x,p: self.h_or_g(x,p)@u
            else:
                raise ValueError('Invalid iEC: iEC = "est" or "con"')
            dW1 = np.random.multivariate_normal(np.zeros(k1),dt*Ik1)*dscale
            x = self.rk4(x,p,dEf,funx)+self.Bw(x,p)@dW1
            xhis[k+1,:] = x
        """
        
        
        2) FIGURE GENERATION
    
        
        """
        print("========================================================")
        print(bNam1)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,np.sum((xhis-zhis)**2,1))
        plt.plot(this,np.ones(np.size(this))*self.Jcv_opt)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(ly,fontsize=FontSize)
        plt.legend([l1,l2],loc="best")
        plt.title(tit1,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam2)
        print("========================================================")
        Nrow = int(self.n/Ncol)+np.remainder(self.n,Ncol)
        fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
        plt.subplots_adjust(wspace=0.25,hspace=0.25)
        if Ncol == 1:
            ax = np.reshape(ax,(self.n,1))
        elif Nrow == 1:
            ax = np.reshape(ax,(1,self.n))
        if xnames == "num":
            xnames = []
            for i in range(self.n):
                xnames += [r"state "+str(i+1)]
        for row in range(Nrow):
            for col in range(Ncol):
                i = Ncol*row+col
                if i+1 <= self.n:
                    ax[row,col].plot(this,xhis[:,i]-zhis[:,i])
                    ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                    if self.iEC == "est":
                        LabelName = r"estimation error: "+xnames[i]
                    elif self.iEC == "con":
                        LabelName = r"tracking error: "+xnames[i]
                    else:
                        txterr = 'Invalid iEC: iEC = "est" or "con"'
                        raise ValueError(txterr)
                    ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
        fig.suptitle(tit2,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print("==================== SIMULATIOM END ====================")
        print("========================================================")
        return this,xhis,zhis
    
    def simulation_OFC(self,sncmE,sncmC,ncmE,ncmC,f,g,Cfun,h,dt,tf,x0,z0,Qe,Qc,Re,Rc,\
        dscale=10,xnames="num",Ncol=1,FigSize=(20,10),FontSize=20,phis=None):
        """
        Perform NCM-based output feedback control of given nolinear dynamical
        systems and return simulation results
        

        Parameters
        ----------
        dt : float
            simulation time step
        tf : float
            terminal time
        x0 : ndarray - (n, )
            initial state
        z0 : ndarray - (n, ), to be used for iEC = "est"
            estimated initial state
        dscale : float, optional, default is 10
            scale of external disturbance 
        xnames : str, optional, default is "num"
            list containing names of each state, when xnames = "num", they are
            denoted as xnames = ["state 1","state 2",...]
        Ncol : int, optional, default is 1
            # columns of state figures to be generated
        FigSize : tuple, optional, default is (20,10)
            size of state figures to be generated
        FontSize : float, optional, default is 20
            font size of figures to be generated
        phis : system parameter history, optional, default is None
            history of system parameters

        Returns
        -------
        this : ndarray - (int(tf/dt)+1, )
            time histry 
        xhis : ndarray - (int(tf/dt)+1,n)
            state history
        zhis : ndarray - (int(tf/dt)+1,n), to be used for estimation tasks
            estimated state history

        """
        """
        
        
        1) SIMULATION
    
        
        """
        if len(sig(f).parameters) == 1:
            fun1 = f
            f = lambda x,p: fun1(x)
        if len(sig(g).parameters) == 1:
            fun2 = g
            g = lambda x,p: fun2(x)
        if len(sig(Cfun).parameters) == 1:
            fun3 = Cfun
            Cfun = lambda x,p: fun3(x)
        if len(sig(h).parameters) == 1:
            fun4 = h
            h = lambda x,p: fun4(x)
        print("========================================================")
        print("====================== SIMULATIOM ======================")
        print("========================================================")
        if dt <= self.dt_rk:
            self.dt_rk = dt
        self.Nrk = int(dt/self.dt_rk)
        Afun5 = lambda x,p: self.jacobian(x,p,self.dynamicsf)
        Cfun5 = lambda x,p: self.jacobian(x,p,self.h_or_g)
        Nsim = int(tf/dt)
        k1 = np.size(ncmC.Bw(self.xlims[0,:],self.plims[0,:]),1)
        k2 = np.size(ncmE.Gw(self.xlims[0,:],self.plims[0,:]),1)
        Ik1 = np.identity(k1)
        Ik2 = np.identity(k2)
        np.set_printoptions(precision=1)
        print("time step =",dt)
        print("terminal time =",tf)
        print("initial state =",x0)
        print("estimated initial state =",z0)
        funx = lambda x,p,dEf: f(x,p)+dEf(x,p)
        z = z0
        zhis = np.zeros((Nsim+1,self.n))
        zhis[0,:] = z
        x = x0
        xhis = np.zeros((Nsim+1,self.n))
        xhis[0,:] = x
        z2 = z0
        z2his = np.zeros((Nsim+1,self.n))
        z2his[0,:] = z2
        x2 = x0
        x2his = np.zeros((Nsim+1,self.n))
        x2his[0,:] = x2
        z3 = z0
        z3his = np.zeros((Nsim+1,self.n))
        z3his[0,:] = z3
        x3 = x0
        x3his = np.zeros((Nsim+1,self.n))
        x3his[0,:] = x3
        z4 = z0
        z4his = np.zeros((Nsim+1,self.n))
        z4his[0,:] = z4
        x4 = x0
        x4his = np.zeros((Nsim+1,self.n))
        x4his[0,:] = x4
        z5 = z0
        z5his = np.zeros((Nsim+1,self.n))
        z5his[0,:] = z5
        x5 = x0
        x5his = np.zeros((Nsim+1,self.n))
        x5his[0,:] = x5
        uehis = np.zeros(Nsim+1)
        ue = 0
        ue2his = np.zeros(Nsim+1)
        ue2 = 0
        ue3his = np.zeros(Nsim+1)
        ue3 = 0
        ue4his = np.zeros(Nsim+1)
        ue4 = 0
        ue5his = np.zeros(Nsim+1)
        ue5 = 0
        tit1 = "Performance of NCM-based Output Feedback (1)"
        tit2 = "Performance of NCM-based Output Feedback (2)"
        tit3 = "Performance of NCM-based Output Feedback (3)"
        tit4 = "Performance of NCM-based Output Feedback (4)"
        tit5 = "Performance of NCM-based Output Feedback (5)"
        ly = r"estimation error: $\|x-\hat{x}\|^2$"
        lyb = r"tracking error: $\|x-x_d\|^2$"
        lyc = r"control effort $\int_0^t\|u(\tau)\|d\tau$"
        l1 = r"estimation error (NCM)"
        l2 = r"estimation error (SDRE)"
        l3 = r"estimation error (NCM)"
        l4 = r"estimation error (CV-STEM)"
        l1b = r"tracking error (SNCM)"
        l2b = r"tracking error (SDRE)"
        l3b = r"tracking error (NCM)"
        l4b = r"tracking error (CV-STEM)"
        bNam1 = "=================== ESTIMATION ERROR ==================="
        bNam2 = "============ ESTIMATION ERROR OF EACH STATE ============"
        bNam3 = "==================== TRACKING ERROR ===================="
        bNam4 = "============= TRACKING ERROR OF EACH STATE ============="
        bNam5 = "==================== CONTROL EFFORT ===================="
        l5 = r"optimal steady-state upper bound"
        dx = 0.01
        d2Mcdx2his = np.zeros(Nsim)
        d2Medx2his = np.zeros(Nsim)
        Mc2his = np.zeros(Nsim)
        Me2his = np.zeros(Nsim)
        if phis == None:
            phis = np.linspace(self.plims[0,:],self.plims[1,:],Nsim)
        for k in range(Nsim):
            p = phis[k,:]
            Ax = self.Afun(z,p)
            Ax2 = self.Afun(z2,p)
            Ax5 = Afun5(z5,p)
            Bx = g(z,p)
            Bx5 = g(z5,p)
            Mc = sncmC.ncm(z,p)
            Mc2 = ncmC.ncm(z3,p)
            sncmE.cvstem0(np.array([z]),np.array([p]),sncmE.alp_opt,sncmE.eps_opt)
            Mecv = np.linalg.inv(sncmE.Ws[0])
            sncmC.cvstem0(np.array([z]),np.array([p]),sncmC.alp_opt,sncmC.eps_opt)
            Mccv = np.linalg.inv(sncmC.Ws[0])
            sncmE.cvstem0(np.array([z4]),np.array([p]),sncmE.alp_opt,sncmE.eps_opt)
            Mecv2 = np.linalg.inv(sncmE.Ws[0])
            sncmC.cvstem0(np.array([z4]),np.array([p]),sncmC.alp_opt,sncmC.eps_opt)
            Mccv2 = np.linalg.inv(sncmC.Ws[0])
            Kc,_,_ = control.lqr(Ax2,g(z2,p),Qc,Rc)
            Kc5,_,_ = control.lqr(Ax5,Bx5,Qc,Rc)
            u = -Bx.T@Mc@z
            u2 = -Kc@z2
            u3 = -g(z3,p).T@Mc2@z3
            u4 = -g(z4,p).T@Mccv2@z4
            u5 = -Kc5@z5
            dEfC = lambda x,p: g(x,p)@u
            dEfC2 = lambda x,p: g(x,p)@u2
            dEfC3 = lambda x,p: g(x,p)@u3
            dEfC4 = lambda x,p: g(x,p)@u4
            dEfC5 = lambda x,p: g(x,p)@u5
            d2Mes = []
            d2Mcs = []
            for i in range(self.n):
                ei = np.zeros(self.n)
                ei[i] = 1
                dxi = ei*dx
                for j in range(self.n):
                    ej = np.zeros(self.n)
                    ej[j] = 1
                    dxj = ej*dx
                    Mdij = sncmE.ncm(z+dxi+dxj,p)
                    Mdi = sncmE.ncm(z+dxi,p)
                    Mdj = sncmE.ncm(z+dxj,p)
                    M = sncmE.ncm(z,p)
                    dMij = (Mdij-Mdi-Mdj+M)/dx**2
                    d2Mes.append(np.linalg.norm(dMij,ord=2))
                    Mdij = sncmC.ncm(z+dxi+dxj,p)
                    Mdi = sncmC.ncm(z+dxi,p)
                    Mdj = sncmC.ncm(z+dxj,p)
                    M = sncmC.ncm(z,p)
                    dMij = (Mdij-Mdi-Mdj+M)/dx**2
                    d2Mcs.append(np.linalg.norm(dMij,ord=2))
            d2Medx2his[k] = max(d2Mes)
            d2Mcdx2his[k] = max(d2Mcs)
            
            dW1 = np.random.multivariate_normal(np.zeros(k1),dt*Ik1)*0.01
            x = self.rk4(x,p,dEfC,funx)+sncmC.Bw(x,p)@dW1
            x2 = self.rk4(x2,p,dEfC2,funx)+ncmC.Bw(x2,p)@dW1
            x3 = self.rk4(x3,p,dEfC3,funx)+ncmC.Bw(x3,p)@dW1
            x4 = self.rk4(x4,p,dEfC4,funx)+ncmC.Bw(x4,p)@dW1
            x5 = self.rk4(x5,p,dEfC5,funx)+ncmC.Bw(x5,p)@dW1
            xhis[k+1,:] = x
            x2his[k+1,:] = x2
            x3his[k+1,:] = x3
            x4his[k+1,:] = x4
            x5his[k+1,:] = x5
            Me = sncmE.ncm(z,p)
            Me2 = ncmE.ncm(z3,p)
            Cx = Cfun(z,p)
            Lx = Me@Cx.T
            Cx2 = Cfun(z2,p)
            Cx5 = Cfun5(z5,p)
            Ke,_,_ = control.lqr(Ax2.T,Cx2.T,Qe,Re)
            Ke5,_,_ = control.lqr(Ax5.T,Cx5.T,Qe,Re)
            Lx2 = Ke.T
            Lx5 = Ke5.T
            Cx3 = Cfun(z3,p)
            Lx3 = Me2@Cx3.T
            Cx4 = Cfun(z4,p)
            Lx4 = Mecv2@Cx4.T
            #Lx = K.T
            dW2 = np.random.multivariate_normal(np.zeros(k2),dt*Ik2)*dscale
            y = h(x,u,p)+sncmE.Gw(x,p)@dW2/dt
            y2 = h(x2,u2,p)+sncmE.Gw(x2,p)@dW2/dt
            y3 = h(x3,u3,p)+sncmE.Gw(x3,p)@dW2/dt
            y4 = h(x4,u4,p)+sncmE.Gw(x4,p)@dW2/dt
            y5 = h(x5,u5,p)+sncmE.Gw(x5,p)@dW2/dt
            funz = lambda z,p,dEf: f(z,p)+g(z,p)@u+dEf(z,p)
            funz2 = lambda z,p,dEf: f(z,p)+g(z,p)@u2+dEf(z,p)
            funz3 = lambda z,p,dEf: f(z,p)+g(z,p)@u3+dEf(z,p)
            funz4 = lambda z,p,dEf: f(z,p)+g(z,p)@u4+dEf(z,p)
            funz5 = lambda z,p,dEf: f(z,p)+g(z,p)@u5+dEf(z,p)
            dEfE = lambda z,p: Lx@(y-h(z,u,p))
            dEfE2 = lambda z,p: Lx2@(y2-h(z,u2,p))
            dEfE3 = lambda z,p: Lx3@(y3-h(z,u3,p))
            dEfE4 = lambda z,p: Lx4@(y4-h(z,u4,p))
            dEfE5 = lambda z,p: Lx5@(y5-h(z,u5,p))
            z = self.rk4(z,p,dEfE,funz)
            z2 = self.rk4(z2,p,dEfE2,funz2)
            z3 = self.rk4(z3,p,dEfE3,funz3)
            z4 = self.rk4(z4,p,dEfE4,funz4)
            z5 = self.rk4(z5,p,dEfE5,funz5)
            zhis[k+1,:] = z
            z2his[k+1,:] = z2
            z3his[k+1,:] = z3
            z4his[k+1,:] = z4
            z5his[k+1,:] = z5
            ue += np.linalg.norm(u)*dt
            ue2 += np.linalg.norm(u2)*dt
            ue3 += np.linalg.norm(u3)*dt
            ue4 += np.linalg.norm(u4)*dt
            ue5 += np.linalg.norm(u5)*dt
            uehis[k+1] = ue
            ue2his[k+1] = ue2
            ue3his[k+1] = ue3
            ue4his[k+1] = ue4
            ue5his[k+1] = ue5
            
            """
            a_e1 = sncmE.Lm*sncmE.b_over**2*(sncmE.eps_opt+1/2)
            a_e2 = sncmE.Lm*sncmE.c_over**2*sncmE.g_over**2*(sncmE.eps_opt+1/2)
            a_gc = sncmC.Lm*sncmC.b_over**2*(sncmC.eps_opt+1/2)
            
            if k == 0:
                Mcp = Mc
                Mep = Me
            else:
                In = np.identity(self.n)
                dMcdt = (Mc-Mcp)/dt
                dMedt = (Me-Mep)/dt
                a_ge = a_e1+sncmE.nu_opt**2*a_e2
                conC = -(dMcdt+Mc@Ax+Ax.T@Mc-2*Mc@Bx@Bx.T@Mc+a_gc*In)
                conE = -(-dMedt+Ax@Me+Me@Ax.T-2*Me@Cx.T@Cx@Me+a_ge*Me@Me)
                lc,vc = np.linalg.eig(conC)
                le,ve = np.linalg.eig(conE)
                print("con: ",np.min(lc))
                print("est: ",np.min(le))
                Mcp = Mc
                Mep = Me
            """  
            #print("test1: ",np.linalg.norm(dEfE(z,p)))
            #print("test2 :",np.linalg.norm(dEfE3(x3,p)))
            Me2his[k] = np.linalg.norm(np.linalg.cholesky(Me)-np.linalg.cholesky(Mecv))**2/sncmE.nu_opt
            Mc2his[k] = np.linalg.norm(np.linalg.cholesky(Mc)-np.linalg.cholesky(Mccv))**2/sncmC.nu_opt
        this = np.linspace(0,tf,Nsim+1)
        """
        
        
        2) FIGURE GENERATION
    
        
        """
        print("========================================================")
        print(bNam1)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,np.sum((xhis-zhis)**2,1))
        plt.plot(this,np.sum((x2his-z2his)**2,1))
        plt.plot(this,np.sum((x3his-z3his)**2,1))
        plt.plot(this,np.sum((x4his-z4his)**2,1))
        plt.plot(this,np.sum((x5his-z5his)**2,1))
        plt.plot(this,np.ones(np.size(this))*sncmE.Jcv_opt)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(ly,fontsize=FontSize)
        plt.legend([l1,l2,l3,l4,l5],loc="best")
        plt.title(tit1,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam2)
        print("========================================================")
        Nrow = int(self.n/Ncol)+np.remainder(self.n,Ncol)
        fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
        plt.subplots_adjust(wspace=0.25,hspace=0.25)
        if Ncol == 1:
            ax = np.reshape(ax,(self.n,1))
        elif Nrow == 1:
            ax = np.reshape(ax,(1,self.n))
        if xnames == "num":
            xnames = []
            for i in range(self.n):
                xnames += [r"state "+str(i+1)]
        for row in range(Nrow):
            for col in range(Ncol):
                i = Ncol*row+col
                if i+1 <= self.n:
                    ax[row,col].plot(this,xhis[:,i]-zhis[:,i])
                    ax[row,col].plot(this,x2his[:,i]-z2his[:,i])
                    ax[row,col].plot(this,x3his[:,i]-z3his[:,i])
                    ax[row,col].plot(this,x4his[:,i]-z4his[:,i])
                    ax[row,col].plot(this,x5his[:,i]-z5his[:,i])
                    ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                    LabelName = r"estimation error: "+xnames[i]
                    ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
                    ax[row,col].legend([r"SNCM",r"SDRE",r"NCM",r"CV-STEM"],loc="best")
        fig.suptitle(tit2,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam3)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,np.sum((xhis)**2,1))
        plt.plot(this,np.sum((x2his)**2,1))
        plt.plot(this,np.sum((x3his)**2,1))
        plt.plot(this,np.sum((x4his)**2,1))
        plt.plot(this,np.sum((x5his)**2,1))
        plt.plot(this,np.ones(np.size(this))*sncmC.Jcv_opt)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(lyb,fontsize=FontSize)
        plt.legend([l1b,l2b,l3b,l4b,l5],loc="best")
        plt.title(tit3,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam4)
        print("========================================================")
        Nrow = int(self.n/Ncol)+np.remainder(self.n,Ncol)
        fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
        plt.subplots_adjust(wspace=0.25,hspace=0.25)
        if Ncol == 1:
            ax = np.reshape(ax,(self.n,1))
        elif Nrow == 1:
            ax = np.reshape(ax,(1,self.n))
        if xnames == "num":
            xnames = []
            for i in range(self.n):
                xnames += [r"state "+str(i+1)]
        for row in range(Nrow):
            for col in range(Ncol):
                i = Ncol*row+col
                if i+1 <= self.n:
                    ax[row,col].plot(this,xhis[:,i])
                    ax[row,col].plot(this,x2his[:,i])
                    ax[row,col].plot(this,x3his[:,i])
                    ax[row,col].plot(this,x4his[:,i])
                    ax[row,col].plot(this,x5his[:,i])
                    ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                    LabelName = r"tracking error: "+xnames[i]
                    ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
                    ax[row,col].legend([r"SNCM",r"SDRE",r"NCM",r"CV-STEM"],loc="best")
        fig.suptitle(tit4,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print(bNam5)
        print("========================================================")
        matplotlib.rcParams.update({"font.size": 15})
        matplotlib.rc("text",usetex=True)
        plt.figure()
        plt.plot(this,uehis)
        plt.plot(this,ue2his)
        plt.plot(this,ue3his)
        plt.plot(this,ue4his)
        plt.plot(this,ue5his)
        plt.xlabel(r"time",fontsize=FontSize)
        plt.ylabel(lyc,fontsize=FontSize)
        plt.legend([r"SNCM",r"SDRE",r"NCM",r"CV-STEM"],loc="best")
        plt.title(tit5,fontsize=FontSize)
        plt.show()
        print("========================================================")
        print("==================== SIMULATIOM END ====================")
        print("========================================================")
        plt.figure()
        plt.plot(this[0:-1],Mc2his)
        plt.plot(this[0:-1],Me2his)
        plt.figure()
        plt.plot(this[0:-1],d2Mcdx2his)
        plt.plot(this[0:-1],sncmC.Lm*np.ones(Nsim))
        plt.figure()
        plt.plot(this[0:-1],d2Medx2his)
        plt.plot(this[0:-1],sncmE.Lm*np.ones(Nsim))
        path = "simdata/"
        np.save(path+"/this.npy",this)
        np.save(path+"/xhis.npy",xhis)
        np.save(path+"/zhis.npy",zhis)
        np.save(path+"/x2his.npy",x2his)
        np.save(path+"/z2his.npy",z2his)
        np.save(path+"/x3his.npy",x3his)
        np.save(path+"/z3his.npy",z3his)
        np.save(path+"/x4his.npy",x4his)
        np.save(path+"/z4his.npy",z4his)
        np.save(path+"/x5his.npy",x5his)
        np.save(path+"/z5his.npy",z5his)
        np.save(path+"/uehis.npy",uehis)
        np.save(path+"/ue2his.npy",ue2his)
        np.save(path+"/ue3his.npy",ue3his)
        np.save(path+"/ue4his.npy",ue4his)
        np.save(path+"/ue5his.npy",ue5his)
        np.save(path+"/d2Mcdx2his.npy",d2Mcdx2his)
        np.save(path+"/d2Medx2his.npy",d2Medx2his)
        np.save(path+"/Mc2his.npy",Mc2his)
        np.save(path+"/Me2his.npy",Me2his)
        np.save(path+"/Jcvc_opt.npy",sncmC.Jcv_opt)
        np.save(path+"/Jcve_opt.npy",sncmE.Jcv_opt)
        np.save(path+"/Lmc.npy",sncmC.Lm)
        np.save(path+"/Lme.npy",sncmE.Lm)
        np.save(path+"/nuc_opt.npy",sncmC.nu_opt)
        np.save(path+"/nue_opt.npy",sncmE.nu_opt)
        np.save(path+"/chic_opt.npy",sncmC.chi_opt)
        np.save(path+"/chie_opt.npy",sncmE.chi_opt)
        return this,xhis,zhis,x3his,z3his,ue3his,phis