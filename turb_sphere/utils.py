import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def integrate_los(f, rprj, rmax_sph):
    """Calculate column density

    Parameters
    ----------
    f : function
        The function that returns the volume density at a given spherical radius.
    rprj : float
        Projected radius at which the line-of-sight integration is performed.
    rmax_sph : float
        The maximum spherical radius to integrate out.

    Returns
    -------
    dcol : float
        Column density.
    """
    def func(z, rprj):
        r = np.sqrt(rprj**2 + z**2)
        return f(r)
    if isinstance(rprj, np.ndarray):
        dcol = []
        for R in rprj:
            zmax = np.sqrt(rmax_sph**2 - R**2)
            res, _ = quad(func, 0, zmax, args=(R,), epsrel=1e-2, limit=200)
            dcol.append(2*res)
        dcol = np.array(dcol)
    else:
        zmax = np.sqrt(rmax_sph**2 - rprj**2)
        res, _ = quad(func, 0, zmax, args=(rprj,), epsrel=1e-2, limit=200)
        dcol = 2*res
    return dcol

def fwhm(f, rmax_sph, which='volume'):
    """Calculate the FWHM of the column density profile

    Parameters
    ----------
    f : function
        The function that returns the volume/column density at a given
        spherical radius.
        Either rho(r) or Sigma(r) depending on `which`.
    rmax_sph : float
        The maximum spherical radius to integrate out.
    which : str, optional
        If the input is volume density, `volume`
        If the input is column density, `column`

    Returns
    -------
    fwhm : float
        The FWHM of the column density profile.
    """
    if which == 'volume':
        dcol0 = integrate_los(f, 0, rmax_sph)
        fwhm = 2*brentq(lambda x: integrate_los(f, x, rmax_sph) - 0.5*dcol0,
                        0, rmax_sph)
    elif which == 'column':
        dcol0 = f(0)
        fwhm = 2*brentq(lambda x: f(x) - 0.5*dcol0, 0, rmax_sph)
    else:
        raise ValueError("which must be either volume or column")
    return fwhm


