#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Dr J Andres Christen, CIMAT-CONAHCYT, Guanajuato, Mexico.

Tool for analyzing Bayesian Linear Models (BLM) with univariate response
and, possibly, correlation in observations.

Uses Cholesky decomposition for optimized calculations.

The implementation follows closely the document:
    
https://arxiv.org/abs/2406.01819

Please use the arXiv preprint to follow this implementation.

"""


from numpy import array, ones, zeros, diagonal, diag, prod, sqrt, exp, log, pi
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg import inv, det
from scipy.stats import norm, multivariate_normal, t, multivariate_t


def CholAndInverse( S, lower=True):
    """Calculate the Cholesky S=LL' and, with foward substitution
    the Cholesky of the inverse inv(S) = A = U U'. 
    Let $\bL \bu_i = \be_i$ then $\bL \bU = \bI$ or
    $\bU = \bL^{-1}$ and note that
    $\bA = (\bL \bL')^{-1} = (\bL^{-1})' (\bL^{-1}) = \bU' \bU$,
    $\bu_i$ are the columns of $\bU$.

    Example:
    
    for n in [4]: #, 50, 100, 200, 300, 400, 500, 1000]:
        L = zeros((n,n))
        for i in range(n):
            L[i,i] = 1 + 0.01*uniform.rvs()
            for j in range(i):
                L[i,j] = 1 + 0.01*uniform.rvs()
        S = L @ L.transpose()
    L, U = CholAndInverse(S)   
    ###       S                        A
    print( L @ L.transpose()  @   U.transpose() @ U )

    U, L = CholAndInverse( S, lower=False)   
    ###           S                        A
    print( U.transpose() @ U   @   L @ L.transpose()   )


   """
    L = cholesky( S, lower=lower)
    n = S.shape[0]
    U = zeros((n,n))
    for i in range(n):
        e = zeros(n)
        e[i] = 1
        U[:,i] = solve_triangular( L, e, lower=lower)
    #A = U.transpose() @ U
    return L, U


def ConstructX( x, p, phi):
    """Construct a design matrix from function phi at design points x:
       n = x.size, X is n x p and the rows of X are constructed
       with the function phi, X[i,:] = phi(x[i]).
    """
    n = x.size
    X = zeros((n,p))
    for i in range(n):
        X[i,:] = phi(x[i])
    return X

def ConstrucT( p, i):
    """Construct a T 1 x p matrix such that th[i] = T @ th""" 
    T = zeros(p)
    T[i] = 1
    return T.reshape((1,p))

def ConstructS( n, x, d, Corr):
    """Construct a n x n correlation matrix: S[i,j] = Corr(d(x[i],x[j]))."""
    S = zeros((n,n))
    for i in range(n):
        for j in range(n):
            S[i,j] = Corr(d(x[i],x[j]))
    return S

def Constructv( x, y, d, Corr):
    """Construct a n x m correlation matrix at x (nx1) and y (mx1) locations:
        v[i,j] = Corr(d(x[i],y[j])).
    """
    n = x.size
    m = y.size
    v = zeros((n,m))
    for i in range(n):
        for j in range(m):
            v[i,j] = Corr(d(x[i],y[j]))
    return v
 

class BLM:
        """
            
        Tool for analyzing Bayesian Linear Models (BLM) with univariate response
        and, possibly, correlation in observations.
        
        Uses Cholesky decomposition for optimized calculations.

        The implementation follows closely the document:
            
        
            
        \bY = \bX \btheta + \bepsilon;
        \bepsilon \sim N( \bzero,  \bSigma/\lambda ).
        \btheta \sim N( \btheta_0, \bA_0/\lambda)
        \lambda \sim Ga( \alpha_0, \beta_0)
        """
        def __init__( self, th0, A0, alpha0, beta0, X, y, Sigma=None, la=None, verbose=True):
            self.th0 = th0 
            self.A0 = A0
            self.alpha0 = alpha0
            self.beta0 = beta0
            self.X = X
            self.p = X.shape[1]
            self.y = y 
            self.n = self.y.shape[0] # sample size
            self.Sigma = Sigma
            self.log_2_pi = log(2*pi)
            if la is not None:
                if verbose:
                    print("Precision (lambda) known.")
                self.la = la
                self.la_known = True
            else:
                self.la_known = False
            if Sigma is None:
                self.uncorr = True
                if verbose:
                    print("Uncorrelated data")
            else:
                if verbose:
                    print("Calculating Cholesky of Sigma")
                self.L, self.U = CholAndInverse(self.Sigma)
                self.uncorr = False
            if verbose:
                print("Calculating posterior parameters")
            if self.uncorr:
                X_t_X = self.X.transpose() @ self.X
                self.An = self.A0 + X_t_X
                self.Un, self.Ln = CholAndInverse( self.An, lower=False)
                self.An_det_sqrt = prod(diagonal(self.Un))
                self.Sigman = self.Ln @ self.Ln.transpose() ### inv(An)
                self.thn = self.Sigman @ (self.A0 @ self.th0 + self.X.transpose() @ self.y )
                dis = self.th0 - self.thn
                self.d2n = (dis.transpose() @ self.A0 @ dis)[0,0]
                self.res = self.y - self.X @ self.thn
                self.s2n = (self.res.transpose() @ self.res)[0,0]
                self.alphan = self.alpha0 + self.n/2
                self.betan = self.beta0 + 0.5*(self.s2n + self.d2n)
                ### The sqrt of the determinant of matrix B_0
                ### For the normalization constant of the model
                self.sqrt_det_B0 = sqrt(det(diag(ones(self.p)) - (self.Ln.transpose() @ X_t_X @ self.Ln)))
            else:
                self.n = self.y.shape[0]
                ### A= U.transpose() @ U # A = U'U
                self.H = self.U @ self.X
                ###                   X' U'U X   = X' A X
                X_t_A_X = self.H.transpose() @ self.H
                ###             A0  +   X' U'U X = An
                self.An = self.A0 + X_t_A_X
                self.Un, self.Ln = CholAndInverse( self.An, lower=False)
                self.An_det_sqrt = prod(diagonal(self.Un))
                self.Sigman = self.Ln @ self.Ln.transpose() ### inv(An)
                ###                          X' U'U y
                self.thn = self.Sigman @\
                    (self.A0 @ self.th0 + self.H.transpose() @ self.U @ self.y )
                dis = self.th0 - self.thn
                self.d2n = (dis.transpose() @ self.A0 @ dis)[0,0]
                self.res = self.U @ (self.y - self.X @ self.thn)
                ###             (y - Xth)'U'U(y - Xth)
                self.s2n = (self.res.transpose() @ self.res)[0,0]
                self.alphan = self.alpha0 + self.n/2
                self.betan = self.beta0 + 0.5*(self.s2n + self.d2n)
                self.sqrt_det_B0 = prod(diagonal(self.U)) *\
                    sqrt(det(diag(ones(self.p)) - self.Ln.transpose() @ X_t_A_X @ self.Ln))
            ### Normalization constant, same expression for both 
            if self.la_known:
                self.norm_const = self.sqrt_det_B0 * exp(-self.la*self.betan)
            else:
                self.norm_const = self.sqrt_det_B0 * (self.betan)**(-self.alphan)

        def LogPost( self, th, la):
            """Log of the Normal-Gamma posterior pdf.
               la is ignored if la_known.
            """
            tmp = self.Un @ (th - self.thn)
            if self.la_known:
                la = self.la
                return -(self.p/2)*self.log_2_pi + (self.p/2)*log(la) + log(self.An_det_sqrt)\
                    -0.5*la* (tmp.transpose() @ tmp)[0,0]
            else:
                return -(self.p/2)*self.log_2_pi + (self.alphan+self.p/2-1)*log(la) + log(self.An_det_sqrt)\
                    -0.5*la* (tmp.transpose() @ tmp)[0,0] -la*self.betan

        def PostMarg( self, T):
            """Calculate a posterior marginal, using projection T."""
            self.marg_k = T.shape[0] ## Dimension of marginal vector T theta
            ### Marginal mean
            self.marg_m = T @ self.thn
            tmp = T @ self.Sigman @ T.transpose()
            if self.la_known:
                ### Variance-covariance matris of the marginal Gaussian
                self.marg_V = tmp/self.la
                if self.marg_k == 1:
                    return norm( loc=self.marg_m[0,0], scale=sqrt(self.marg_V[0,0]))
                else:
                    return multivariate_normal( loc=self.marg_m.flatten(), cov=self.marg_V)
            else:
                ### Dispersion matrix of marginal t
                self.marg_D = (self.betan/self.alphan) * tmp
                self.marg_nu = 2*self.alphan
                if self.marg_k == 1:
                    return t( df=self.marg_nu, loc=self.marg_m[0,0], scale=sqrt(self.marg_D[0,0]))
                else:
                    return multivariate_t( df=self.marg_nu, loc=self.marg_m.flatten(), shape=self.marg_D)
            
        def Pred( self, Xm, Sm=None, vm=None):
            """Calculate the predictive distribution at Xm m x p matrix.
               $V(\bZ) = \lambda^{-1}\bS^m$ the $m \times m$
               variance-covariance matrix of $\bZ$ and
               $cov( \bY, \bZ) = \lambda^{-1} \bv_m$, the cross covariances.          
            """
            self.pred_m = Xm.shape[0]
            if self.uncorr:
                self.pred_mu = Xm @ self.thn
                tmp  = diag(ones(self.pred_m)) +\
                    Xm @ (self.Ln @ self.Ln.transpose()) @ Xm.transpose()
            else:
                a = self.U @ vm
                # H = U X, a = U vm, h = (Xm - vm'AX)Ln
                h = (Xm - a.transpose() @ self.H) @ self.Ln
                self.pred_mu = Xm @ self.thn + a.transpose() @ self.res
                tmp = Sm - a.transpose() @ a + h @ h.transpose()
            if self.la_known:
                ### Variance-covariance matrix of the marginal Gaussian
                self.pred_V = tmp/self.la
                if self.pred_m == 1:
                    return norm( loc=self.pred_mu[0,0], scale=sqrt(self.pred_V[0,0]))
                else:
                    return multivariate_normal( loc=self.pred_mu, cov=self.pred_V)
            else:
                ### Dispersion matrix of marginal t
                self.pred_D = (self.betan/self.alphan) * tmp
                self.pred_nu = 2*self.alphan
                if self.pred_m == 1:
                    return t( df=self.pred_nu, loc=self.pred_mu[0,0], scale=sqrt(self.pred_D[0,0]))
                else:
                    return multivariate_t( df=self.pred_nu, loc=self.pred_m, shape=self.pred_D)
              
        















