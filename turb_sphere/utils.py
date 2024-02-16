import numpy as np
from scipy.integrate import quad, dblquad
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


def integrate_2d_projected(f, rmax_prj, rmax_sph):
    """Calculate 2D integral over the projected area

    Parameters
    ----------
    f : function
        The function that returns the volume density at a given spherical radius.
    rmax_prj : float
        Maximum radial extent in the projected plane.
    rmax_sph : float
        The maximum spherical radius to integrate out.

    Returns
    -------
    dcol : float
        Column density.
    """
    def zmax(rprj):
        return np.sqrt(rmax_sph**2 - rprj**2)

    def func(z, rprj):
        r = np.sqrt(rprj**2 + z**2)
        return f(r)*2*np.pi*rprj

    res, _ = dblquad(func, 0, rmax_prj, lambda x: 0, zmax, epsrel=1e-2)
    return 2*res


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
