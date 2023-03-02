from ._distn_infrastructure import (
    rv_continuous
)

import numpy as np
import warnings 


__all__ = ['alpha','anglit']

class alpha_gen(rv_continuous):
     r"""
     An alpha continuous random variable.
     
     """
     
     def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

     def _pdf(self, x, a):
        # alpha.pdf(x, a) = 1/(x**2*Phi(a)*sqrt(2*pi)) * exp(-1/2 * (a-1/x)**2)
        return 1.0/(x**2)/_norm_cdf(a)*_norm_pdf(a-1.0/x)

     def _logpdf(self, x, a):
        return -2*np.log(x) + _norm_logpdf(a-1.0/x) - np.log(_norm_cdf(a))
    
     def _cdf(self, x, a):
        return _norm_cdf(a-1.0/x) / _norm_cdf(a)
    
     def _stats(self, a):
        return [np.inf]*2 + [np.nan]*2
     
alpha = alpha_gen(a=0.0, name='alpha')


class anglit_gen(rv_continuous):
    
    r"""
    An anglit continuous random variable.
    
    """
    def _shape_info(self):
        return []

    def _pdf(self, x):
        # anglit.pdf(x) = sin(2*x + \pi/2) = cos(2*x)
        return np.cos(2*x)

    def _cdf(self, x):
        return np.sin(x+np.pi/4)**2.0

    def _ppf(self, q):
        return np.arcsin(np.sqrt(q))-np.pi/4

    def _stats(self):
        return 0.0, np.pi*np.pi/16-0.5, 0.0, -2*(np.pi**4 - 96)/(np.pi*np.pi-8)**2

    def _entropy(self):
        return 1-np.log(2)
    
    
anglit = anglit_gen(a=-np.pi/4, b=np.pi/4, name='anglit')



class arcsine_gen(rv_continuous):
    r"""An arcsine continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density function for `arcsine` is:
    .. math::
        f(x) = \frac{1}{\pi \sqrt{x (1-x)}}
    for :math:`0 < x < 1`.
    %(after_notes)s
    %(example)s
    """
    def _shape_info(self):
        return []

    def _pdf(self, x):
        # arcsine.pdf(x) = 1/(pi*sqrt(x*(1-x)))
        with np.errstate(divide='ignore'):
            return 1.0/np.pi/np.sqrt(x*(1-x))

    def _cdf(self, x):
        return 2.0/np.pi*np.arcsin(np.sqrt(x))

    def _ppf(self, q):
        return np.sin(np.pi/2.0*q)**2.0

    def _stats(self):
        mu = 0.5
        mu2 = 1.0/8
        g1 = 0
        g2 = -3.0/2.0
        return mu, mu2, g1, g2

    def _entropy(self):
        return -0.24156447527049044468


arcsine = arcsine_gen(a=0.0, b=1.0, name='arcsine')





class beta_gen(rv_continuous):
    r"""A beta continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density function for `beta` is:
    .. math::
        f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}
    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).
    `beta` takes :math:`a` and :math:`b` as shape parameters.
    %(after_notes)s
    %(example)s
    """
    def _shape_info(self):
        ia = _ShapeInfo("a", False, (0, np.inf), (False, False))
        ib = _ShapeInfo("b", False, (0, np.inf), (False, False))
        return [ia, ib]

    def _rvs(self, a, b, size=None, random_state=None):
        return random_state.beta(a, b, size)

    def _pdf(self, x, a, b):
        #                     gamma(a+b) * x**(a-1) * (1-x)**(b-1)
        # beta.pdf(x, a, b) = ------------------------------------
        #                              gamma(a)*gamma(b)
        return _boost._beta_pdf(x, a, b)

    def _logpdf(self, x, a, b):
        lPx = sc.xlog1py(b - 1.0, -x) + sc.xlogy(a - 1.0, x)
        lPx -= sc.betaln(a, b)
        return lPx

    def _cdf(self, x, a, b):
        return _boost._beta_cdf(x, a, b)

    def _sf(self, x, a, b):
        return _boost._beta_sf(x, a, b)

    def _isf(self, x, a, b):
        with warnings.catch_warnings():
            # See gh-14901
            message = "overflow encountered in _beta_isf"
            warnings.filterwarnings('ignore', message=message)
            return _boost._beta_isf(x, a, b)
        
        
        
beta = beta_gen(a=0.0, b=1.0, name='beta')

















class rv_histogram(rv_continuous):
    def __init__(self) -> None:
        super().__init__()
        
        
        
        
class argus_gen(rv_continuous):
    r"""
    Argus distribution
    References
    ----------
    .. [1] "ARGUS distribution",
           https://en.wikipedia.org/wiki/ARGUS_distribution
    """
    def _shape_info(self):
        return [_ShapeInfo("chi", False, (0, np.inf), (False, False))]

    def _logpdf(self, x, chi):
        # for x = 0 or 1, logpdf returns -np.inf
        with np.errstate(divide='ignore'):
            y = 1.0 - x*x
            A = 3*np.log(chi) - _norm_pdf_logC - np.log(_argus_phi(chi))
            return A + np.log(x) + 0.5*np.log1p(-x*x) - chi**2 * y / 2

    def _pdf(self, x, chi):
        return np.exp(self._logpdf(x, chi))

    def _cdf(self, x, chi):
        return 1.0 - self._sf(x, chi)

    def _sf(self, x, chi):
        return _argus_phi(chi * np.sqrt(1 - x**2)) / _argus_phi(chi)