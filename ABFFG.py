# Automatic backward filtering and forward guiding
import jax
import jax.numpy as jnp

from SDE import dot, solve, forward

# Quadratic form: xᵀ H x
quadratic = lambda x, H: jnp.dot(x, jnp.dot(H, x))

# Standard Gaussian log-density with covariance matrix Σ
logphi = lambda x, mu, Sigma: jax.scipy.stats.multivariate_normal.logpdf(x, mu, Sigma)

# Standard Gaussian PDF
phi = lambda x, mu, Sigma: jax.scipy.stats.multivariate_normal.pdf(x, mu, Sigma)

# Normalization constant: (2π)^(-d/2) * det(Σ)^(-1/2)
omega = lambda Sigma: (jnp.linalg.det(Sigma) * (2 * jnp.pi)**Sigma.shape[0])**(-0.5)

# Normalization constant using precision matrix H = Σ⁻¹
omega_H = lambda H: jnp.sqrt(jnp.linalg.det(H) / (2 * jnp.pi)**H.shape[0])

# Log of normalization constant with Σ
logomega = lambda Sigma: -0.5 * (jnp.linalg.slogdet(Sigma)[1] + jnp.log(2 * jnp.pi) * Sigma.shape[0])

# Log of normalization constant with H
logomega_H = lambda H: 0.5 * (jnp.linalg.slogdet(H)[1] - jnp.log(2 * jnp.pi) * H.shape[0])

# Log-density with precision matrix: log 𝒩(x | μ, H)
logphi_H = lambda x, mu, H: logomega_H(H) - 0.5 * quadratic(x - mu, H)

# Density with precision matrix
phi_H = lambda x, mu, H: jnp.exp(logphi_H(x, mu, H))

# Log-density in canonical form: log 𝒩(y | H⁻¹F, H)
logphi_can = lambda y, F, H: logphi_H(y, jnp.linalg.solve(H, F), H)

# PDF in canonical form
phi_can = lambda y, F, H: jnp.exp(logphi_can(y, F, H))

# Unnormalized log-density in canonical form
logU = lambda y, c, F, H: c - 0.5 * quadratic(y, H) + jnp.dot(F, y)

# Unnormalized density
U = lambda y, c, F, H: jnp.exp(logU(y, c, F, H))

# Forward guided sampling, assumes already backward filtered (H,F parameters)
def forward_guided(x,H_T,F_T,tildea,dts,dWs,b,sigma,params):
    tildebeta = lambda t,params: 0.
    tildeb = lambda t,x,params: tildebeta(t,params) #+jnp.dot(tildeB,x) #tildeB is zero for now

    T = jnp.sum(dts)
    Phi_inv = lambda t: jnp.eye(H_T.shape[0])+H_T@tildea*(T-t)
    Ht = lambda t: solve(Phi_inv(t),H_T).reshape(H_T.shape) 
    Ft = lambda t: solve(Phi_inv(t),F_T).reshape(F_T.shape) 

    def bridge_SFvdM(carry, val):
        t, X, logpsi = carry
        #dt, dW, H, F = val
        dt, dW = val
        H = Ht(t); F = Ft(t)
        tilderx =  F-dot(H,X)
        _sigma = sigma(x,params)
        _a = jnp.einsum('ij,kj->ik',_sigma,_sigma)
        n = _a.shape[0]
        
        # SDE
        Xtp1 = X + b(t,X, params)*dt + dot(_a,tilderx)*dt + dot(_sigma,dW)
        tp1 = t + dt
        
        # Logpsi
        amtildea = _a-tildea
        logpsicur = logpsi+(
                jnp.dot(b(t,X,params)-tildeb(t,X,params),tilderx)
                -.5*jnp.einsum('ij,ji->',amtildea,H)
                +.5*jnp.einsum('ij,jd,id->',
                           amtildea,tilderx.reshape((n,-1)),tilderx.reshape((n,-1)))
                    )*dt
        return((tp1,Xtp1,logpsicur),(t,X,logpsi))    

    # Sample
    (T,X,logpsi), (ts,Xs,logpsis) = jax.lax.scan(bridge_SFvdM,(0.,x,0.),(dts,dWs))#,H,F))
    Xscirc = jnp.vstack((Xs, X))
    return Xscirc,logpsi