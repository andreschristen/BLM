#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Dr J Andres Christen, CIMAT-CONAHCYT, Guanajuato, Mexico.

Examples of BLM, pyblm

Tool for analyzing Bayesian Linear Models (BLM) with univariate response
and, ppssibly, correlation in observations.

Uses Cholesky decomposition for optimized calculations.

Please read the accompanying document:

https://arxiv.org/abs/2406.01819

03 June 2024

"""

from numpy import zeros, array, linspace, diag, ones, tri, inner, sin, pi, exp
from scipy.stats import norm, multivariate_normal
from matplotlib.pylab import subplots

from pyblm import BLM, ConstructX, ConstrucT, ConstructS, Constructv
from plotfrozen import PlotFrozenDist


def VerySimpleExample():
    """Data simlated from:
        x = linspace( 0, 10, num=21)
        sigma = 1
        y = 0.3*x**2 + 0.1*x + 1 + sigma*norm.rvs(size=21)
     """   
    n=21
    ### some data
     
    y = array([-0.59209838,  0.62425791,  1.471196  ,  1.85524359,  2.22954708,
                3.85303196,  4.637359  ,  7.24276395,  6.6645275 ,  7.24332852,
                9.96270966,  8.14657624, 12.86083692, 14.62652484, 16.1448467 ,
               18.39791899, 19.13655693, 21.27566144, 24.28036618, 29.52014624,
               33.20452348])    
    y = y.reshape((n,1))
    # observed at the points
    x = array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,
            5.5,  6. ,  6.5,  7. ,  7.5,  8. ,  8.5,  9. ,  9.5, 10. ])
    
    ### Desing matrix
    p=3 # a constant a two covariates, x and x**2
    X = zeros((n,p))
    X[:,0] = ones(n) # constant
    X[:,1] = x # Linear term
    X[:,2] = x**2 #Cuadratic term
    
    ### Prior
    th0 = zeros((p,1))
    A0 = 0.001*diag(ones(3)) # Uninformative
    alpha0 = 1
    beta0 = 1
    
    ###  This opens a Bayesian Linear Model instance
    ### Uncorrelated data, Sigma=None, la (precison par) unnkown
    blm = BLM( th0=th0, A0=A0, alpha0=alpha0, beta0=beta0,\
              X=X, y=y, Sigma=None, la=None) 
    
    ### Make some plots
    fig, ax = subplots(nrows=2,ncols=2)
    ax = ax.flatten()
    ### Plot postrior marginals of each parameter:
    ### We have 3 parameters.  The posterior marginals of each are:
    for i in range(3):
        ### ConstrucT( 3, 0) produces array([[1., 0., 0.]]), the matrix T as in the document
        ### to extract the marginal of the first coeficient etc 
        pmarg = blm.PostMarg(ConstrucT( 3, i))
        PlotFrozenDist( pmarg, ax=ax[i])
        ax[i].set_xlabel(r"Posterior t marginal for $\theta_%d$" % (i,))
        ax[i].set_ylabel("Density")
    ax[3].scatter( x, y, s=5) # data points
    ax[3].plot( x, X @ blm.thn, 'r-') #Posterior mean
    ### Produce some predictive quantiles including extrapolation
    Qpred = zeros((25,4)) # matrix of quantiles
    Xm = zeros((1, 3))
    z = linspace( 0, 12, num=25) # Predict at these points
    for i in range(25):
        Xm[0, :] = [ 1, z[i], z[i]**2] # The matrix Xm, as in the document
        pred = blm.Pred(Xm = Xm) # The resulting predictive distribution at z[i]
        Qpred[i,:] = array([pred.ppf(0.1),pred.ppf(0.25),pred.ppf(0.75),pred.ppf(0.9)])
    ax[3].fill_between( z, Qpred[:,0], Qpred[:,3], color="blue", alpha=0.25)
    ax[3].fill_between( z, Qpred[:,1], Qpred[:,2], color="blue", alpha=0.25)
    ax[3].set_xlabel(r"$x$")
    ax[3].set_ylabel(r"Response")
    fig.tight_layout()
    


def ExampleRegression(la_known=False):
    """The sin regression example in the document."""
    #### The true regressor
    F = lambda x: 1 + sin(2*pi*x) ##True regressor
    x0, x1 = 0, 1
    sigma = 0.1
    
    """    #Sigma = 0.5*(tri( n,n,k=1) * tri( n,n,k=1).transpose()) + 0.5*diag(ones(n))

    ###
    th_true = array([ 2, 3, 3, -0.2, 0.003])
    F = lambda x: inner(array([ 1, x, x**2, x**3, x**4]), th_true) ##True regressor   
    x0, x1 = -5, 20 #design interval
    sigma=30 #std. error
    """
    if la_known:
        la = sigma**-2
    else:
        la = None
    
    ### Data simulation
    n = 40 # sample size
    m_true = zeros((n,1))
    x = linspace( x0, x1, num=n) # design, where F is measured
    for i in range(n):
        m_true[i,0] = F(x[i]) #True mean for data
    Sigma=None #cov matrix, of None = I
    ### Data simulation
    y = zeros((n,1))
    if Sigma is None:
        y = m_true + sigma*norm.rvs(size=(n,1))
    else:
        y = m_true + multivariate_normal(cov=sigma**2 * Sigma).rvs().reshape((n,1))

    ### Regression
    fig, ax = subplots(ncols=2,nrows=3, sharex=True) # fit
    ax = ax.flatten()
    fig2, ax2 = subplots(ncols=2,nrows=3) #Posterios
    ax2 = ax2.flatten()
    norm_const = []
    ps = array([1,2,3,4,5,6])
    for p in ps:
        #p  #Number of regression parameters
        #To construct design matrix X
        phi = lambda xi: array([xi]*p)**array(range(p)) #p=4 array([ 1, xi, xi**2, xi**3])
        X = ConstructX( x, p, phi)
        ### Prior parameters
        th0 = zeros((p,1))
        A0 = 0.001*diag(ones(p))
        alpha0 = 1
        beta0 = 1
        blm = BLM( th0=th0, A0=A0, alpha0=alpha0, beta0=beta0,\
                  X=X, y=y, Sigma=Sigma, la=la)
        norm_const += [blm.norm_const]
        ax[p-1].plot( x, y, 'k.')
        ax[p-1].plot( x, m_true, 'k-')
        ax[p-1].plot( x, X @ blm.thn, 'b-')
        for i in range(p):
            marg = blm.PostMarg(T=ConstrucT(p,i)) #univariate t dist
            PlotFrozenDist( marg, color="darkblue", alpha=p/max(ps), ax=ax2[i])
            ax2[i].set_xlabel(r"$\theta_%d$" % (i,))
    fig.tight_layout()
    fig.savefig("../Doc/Fit.png")
    fig2.tight_layout()
    fig2.savefig("../Doc/Posts.png")
        
    ### Model comparisons    
    norm_const = array(norm_const)
    model_post = norm_const/sum(norm_const)
    print(model_post)
    fig, ax = subplots()
    ax.plot( ps, model_post, '-o')
    ax.set_xlabel(r"$p$")
    fig.savefig("../Doc/ModelPost.png")

    """
    ### Test the marginal distributions
    ### Univariate t distibution of parameter     
    marg = blm.PostMarg(T=ConstrucT(p,3))
    #PlotFrozenDist(marg) #plot is, simulate from it rvs etc.
    ### marginal multivariate t distribution of all theta's
    margall = blm.PostMarg(T=diag(ones(p)))
    ### simulate from it and        sum all
    ### sample from the marginal posterior of the sum(th)
    samples = margall.rvs(size=1000) @ ones(p).reshape((p,1))
    ### univariate t of the sum of all parameters
    margs = blm.PostMarg(T=ones(p).reshape((1,p)))
    ### Compare both 
    ax = PlotFrozenDist(margs)
    ax.hist( samples, bins=20, density=True)
    """
    
    ### Test the predictive
    fig, ax = subplots()
    ax.plot( x, y, 'k.')
    ax.plot( x, m_true, 'k-')
    ax.plot( x, X @ blm.thn, 'b-')
    pred = blm.Pred(Xm = phi(0.4).reshape((1,p)))
    #PlotFrozenDist(pred)
    ax.vlines( 0.4, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( 0.4, pred.ppf(0.25), pred.ppf(0.75), color="blue")
    ### Extrapolate
    pred = blm.Pred(Xm = phi(1.01).reshape((1,p)))
    ax.vlines( 1.01, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( 1.01, pred.ppf(0.25), pred.ppf(0.75), color="blue")
    fig.savefig("../Doc/Pred.png")

    return blm




def GaussianCorr( d, s):
    """This is the Gaussina correlation function.  A Matern will be more common."""
    return exp(-0.5*(d/s)**2)

def ExampleGP():
    """Example using a Gaussian Process."""
    #### The true regressor
    F = lambda x: exp(-x)*sin(2*pi*x) ##True regressor
    x0, x1 = 0, 6
    sigma = 0.001
        
    ### Data simulation
    n = 50 # sample size
    m_true = zeros((n,1))
    x = linspace( x0, x1, num=n) # design, where F is measured
    for i in range(n):
        m_true[i,0] = F(x[i]) #True mean for dataself.U @

    Corr = lambda d: GaussianCorr(d, s=0.5)
    Corr = lambda d: exp(-d/10)    
    d = lambda xi,xj: abs(xi-xj)
    Sigma=ConstructS( n, x, d, Corr) #cov matrix
    ### Data simulation
    y = zeros((n,1))
    if Sigma is None:
        y = m_true + sigma*norm.rvs(size=(n,1))
    else:
        y = m_true + multivariate_normal(cov=sigma**2 * Sigma).rvs().reshape((n,1))

    ### Regression
    p=1
    #p  #Number of regression parameters
    #To construct design matrix X
    phi = lambda xi: xi #p=4 array([ 1, xi, xi**2, xi**3])
    X = ConstructX( x, p, phi)
    ### Prior parameters
    th0 = zeros((p,1))
    A0 = 0.001*diag(ones(p))
    alpha0 = 1
    beta0 = 1
    gp = BLM( th0=th0, A0=A0, alpha0=alpha0, beta0=beta0,\
              X=X, y=y, Sigma=Sigma, la=None)
    fig, ax = subplots()
    ax.plot( x, m_true, 'k-')
    ax.plot( x, y, 'k.')
    ax.plot( x, X @ gp.thn, 'b-')

    xi = x[10] - 0.00001
    Sm=array([1]).reshape(1,1)
    vm = Constructv( x, array([xi]), d, Corr)
    pred = gp.Pred(Xm = phi(xi).reshape((1,p)), Sm=Sm, vm=vm)
    #PlotFrozenDist(pred)
    ax.vlines( xi, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( xi, pred.ppf(0.25), pred.ppf(0.75), color="blue")

    xi = x[10] - 0.01
    Sm=array([1]).reshape(1,1)
    vm = Constructv( x, array([xi]), d, Corr)
    pred = gp.Pred(Xm = phi(xi).reshape((1,p)), Sm=Sm, vm=vm)
    #PlotFrozenDist(pred)
    ax.vlines( xi, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( xi, pred.ppf(0.25), pred.ppf(0.75), color="blue")

    xi = x[9] + 0.05
    Sm=array([1]).reshape(1,1)
    vm = Constructv( x, array([xi]), d, Corr)
    pred = gp.Pred(Xm = phi(xi).reshape((1,p)), Sm=Sm, vm=vm)
    #PlotFrozenDist(pred)
    ax.vlines( xi, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( xi, pred.ppf(0.25), pred.ppf(0.75), color="blue")

    xi = x[9] + 0.01
    Sm=array([1]).reshape(1,1)
    vm = Constructv( x, array([xi]), d, Corr)
    pred = gp.Pred(Xm = phi(xi).reshape((1,p)), Sm=Sm, vm=vm)
    #PlotFrozenDist(pred)
    ax.vlines( xi, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( xi, pred.ppf(0.25), pred.ppf(0.75), color="blue")

    xi = x[9] + 0.00001
    Sm=array([1]).reshape(1,1)
    vm = Constructv( x, array([xi]), d, Corr)
    pred = gp.Pred(Xm = phi(xi).reshape((1,p)), Sm=Sm, vm=vm)
    #PlotFrozenDist(pred)
    ax.vlines( xi, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( xi, pred.ppf(0.25), pred.ppf(0.75), color="blue")

    xi = x[9] - 0.01
    Sm=array([1]).reshape(1,1)
    vm = Constructv( x, array([xi]), d, Corr)
    pred = gp.Pred(Xm = phi(xi).reshape((1,p)), Sm=Sm, vm=vm)
    #PlotFrozenDist(pred)
    ax.vlines( xi, pred.ppf(0.1), pred.ppf(0.9), color="lightblue")
    ax.vlines( xi, pred.ppf(0.25), pred.ppf(0.75), color="blue")

    fig.savefig("../Doc/GP1.png")
    ax.set_xlim((1.0,1.4))
    ax.set_ylim((0.0,0.4))
    fig.savefig("../Doc/GP2.png")

    return gp


if __name__ == "__main__":

    VerySimpleExample()
    #blm = ExampleRegression(la_known=False)
    #gp  = ExampleGP()
    