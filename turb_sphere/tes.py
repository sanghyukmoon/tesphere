from scipy.integrate import solve_ivp, quad, odeint
from scipy.optimize import brentq, newton
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle
from pathlib import Path


class TES:
    """Turbulent equilibrium sphere of a fixed central density.

    Parameters
    ----------
    pindex : float, optional
        power-law index of the velocity dispersion.
    rsonic : float, optional
        dimensionless sonic radius.

    Attributes
    ----------
    pindex : float
        power-law index of the velocity dispersion.
    rsonic : float
        dimensionless sonic radius.
    """
    def __init__(self, pindex=0.5, rsonic=np.inf, compute_rcrit=True):
        self._rfloor = 1e-5
        self._sigma = None
        self.pindex = pindex
        self.rsonic = rsonic
        if compute_rcrit:
            self.rcrit = self.critical_radius()
            self.mcrit = self.enclosed_mass(self.rcrit)
            self.ucrit, _ = self.solve(self.rcrit)

    def density(self, r):
        """Calculate normalized density.

        Notes
        -----
        rho/rho_c = exp(-u)
        """
        u, _ = self.solve(r)
        return np.exp(-u)

    def velocity_dispersion(self, r):
        return (r / self.rsonic)**(self.pindex)

    def enclosed_mass(self, xi0):
        """Calculates dimensionless enclosed mass.

        The dimensionless mass enclosed within the dimensionless radius xi
        is defined by
            M(xi) = m(xi)M_{J,c}
        where M_{J,c} is the Jeans mass at the central density rho_c

        Parameters
        ----------
        xi0 : float
            Radius within which the enclosed mass is computed. If None, use
            the maximum radius of a sphere.

        Returns
        -------
        float
            Dimensionless enclosed mass.
        """
        # If xi0 is inf, mass is also inf.
        if isinstance(xi0, float) and np.isinf(xi0):
            return np.inf
        else:
            mask = np.isinf(xi0)
        u, du = self.solve(xi0)
        chi = self.chi(xi0)
        dchi = 2*self.pindex*(chi - 1)/xi0
        m = xi0**2*(chi*du - dchi)
        if not isinstance(xi0, float):
            m[mask] = np.inf
        # return scala when the input is scala
        return m.squeeze()[()]

    def critical_radius(self):
        """Find critical TES radius

        Returns
        -------
        float
            Critical radius
        """
        xi = np.logspace(0, 3, 512)
        kappa = self.bulk_modulus(xi)
        idx = (kappa < 0).nonzero()[0]
        if len(idx) < 1:
            # kappa is everywhere positive, meaning that this TES
            # is stable at every radius. Return inf.
            return np.inf
        else:
            idx = idx[0] - 1
        func = lambda x: self.bulk_modulus(10**x)

        x0, x1 = np.log10(xi[idx]), np.log10(xi[idx+1])
        try:
            logrmax = newton(func, x0, x1=x1).squeeze()[()]
        except:
            logrmax = np.nan
        return 10**logrmax

    def bulk_modulus(self, xi):
        # Perturbation with the fixed turbulent velocity field
        # i.e., p = r_s = const.
        u, du = self.solve(xi)
        m = self.enclosed_mass(xi)
        chi = self.chi(xi)
        dsu, dslogm = self._get_sonic_radius_derivatives(xi)
        dslogchi = -2*self.pindex*(1 - 1/chi)
        num = 1 - 0.5*dsu + 0.5*dslogchi\
                - m**2/(2*chi*np.exp(-u)*xi**4)*(1 - dslogm)
        denom = 1 - m/(xi**3*np.exp(-u))*(1 - dslogm)
        kappa = (2/3)*num/denom
        return kappa

    def chi(self, r):
        """Dimensionless function chi = 1 + (dv/cs)^2

        Equivalent to the ratio of the total to thermal pressure.
        """
        return 1 + self.velocity_dispersion(r)**2

    @property
    def sigma(self):
        """Compute or return the velocity dispersion"""
        if self._sigma is None:
            self._sigma = self.compute_sigma()
        return self._sigma

    def compute_sigma(self):
        """Calculate mass-weighted mean velocity dispersion within rcrit

        Returns
        -------
        sigv : float
            Mass-weighted radial velocity dispersion
        """
        def func(r):
            rho = self.density(r)
            sigma2 = self.velocity_dispersion(r)**2
            return r**2*rho*sigma2
        num, _ = quad(func, self._rfloor, self.rcrit)

        def func(r):
            rho = self.density(r)
            return r**2*rho
        den, _ = quad(func, self._rfloor, self.rcrit)

        sigv = np.sqrt(num/den)
        return sigv

    def solve(self, r):
        """Solve equilibrium equation

        Parameters
        ----------
        r : array
            Dimensionless radii

        Returns
        -------
        u : array
            Log density u = log(rho/rho_c)
        du : array
            Derivative of u: d(u)/d(r)
        """
        if isinstance(r, float) and np.isinf(r):
            return np.inf, 0
        else:
            mask = np.isinf(r)

        r = np.array(r, dtype='float64')
        y0 = np.array([0, 0])
        if r.min() > self._rfloor:
            r = np.insert(r, 0, self._rfloor)
            istart = 1
        else:
            istart = 0
        if np.all(r <= self._rfloor):
            u = y0[0]*np.ones(r.size)
            du = y0[1]*np.ones(r.size)
        else:
            y = odeint(self._dydx, y0, np.log(r),
                       Dfun=self._jac, col_deriv=True)
            u = y[istart:, 0]
            du = y[istart:, 1]/r[istart:]

        if not isinstance(r, float):
            u[mask] = np.inf
            du[mask] = 0

        # return scala when the input is scala
        return u.squeeze()[()], du.squeeze()[()]

    def _dydx(self, y, t):
        """Differential equation for hydrostatic equilibrium.

        Parameters
        ----------
        y : array
            Vector of dependent variables
        x : array
            Independent variable

        Returns
        -------
        array
            Vector of (dy/dx)
        """
        y1, y2 = y
        dy1 = y2
        x = np.exp(t)

        chi = self.chi(x)
        dchi = 2*self.pindex*(chi - 1)
        ddchi = 2*self.pindex*dchi

        a = chi
        b = chi + dchi
        c = -dchi - ddchi - np.exp(2*t - y1)
        dy2 = -(b/a)*y2 - (c/a)
        return np.array([dy1, dy2])

    def _jac(self, y, t):
        y1, y2 = y
        x = np.exp(t)
        chi = self.chi(x)
        dchi = 2*self.pindex*(chi - 1)
        j11 = 0
        j12 = 1
        j21 = -(1./chi)*np.exp(2*t - y1)
        j22 = -(1 + dchi/chi)
        return [[j11, j21], [j12, j22]]

    def _get_sonic_radius_derivatives(self, xi, dlog_xi_s=1e-3):
        log_xi_s = np.log(self.rsonic)

        rsonic = [np.exp(log_xi_s - dlog_xi_s), np.exp(log_xi_s + dlog_xi_s)]
        tsc = [TES(pindex=self.pindex, rsonic=rsonic[0], compute_rcrit=False),
               TES(pindex=self.pindex, rsonic=rsonic[1], compute_rcrit=False)]
        u = [tsc[0].solve(xi)[0], tsc[1].solve(xi)[0]]
        m = [tsc[0].enclosed_mass(xi), tsc[1].enclosed_mass(xi)]
        du = (u[1] - u[0]) / (2*dlog_xi_s)
        dlogm = (np.log(m[1]) - np.log(m[0])) / (2*dlog_xi_s)

        return du, dlogm


class Logotrope:
    """Turbulent equilibrium sphere of a fixed central density.

    xi = r/r_0 where r_0 = 3c_s/np.sqrt(4 pi G rho_c) following
    McLaughlin & Pudritz (1996). This scale length is three times
    the BE scale length.
    """
    def __init__(self, amp):
        """
        Parameters
        ----------
        amp : A_MP the parameter A of McLaughlin & Pudritz
        """
        self._rfloor = 1e-5
        self._sigma = None
        self.amp = amp
        self.rcrit = self.critical_radius()
        self.mcrit = self.enclosed_mass(self.rcrit)

    def density(self, r):
        """Calculate normalized density.

        Notes
        -----
        rho = rho_c * e^u
        """
        u, _ = self.solve(r)
        return np.exp(-u)

    def velocity_dispersion(self, r):
        r = np.atleast_1d(r)
        u, _ = self.solve(r)
        res = np.zeros(len(r))
        sig2 = (1 - self.amp*u)*np.exp(u) - 1
        np.sqrt(sig2, where=(sig2 >= 0), out=res)
        return res

    def enclosed_mass(self, r):
        """
        M = (c_s^2 * r_0 / G) * m
        With this definition, rho_avg/rho_c = m / (3*r^3)
        """
        # If xi0 is inf, mass is also inf.
        if isinstance(r, float) and np.isinf(r):
            return np.inf
        else:
            mask = np.isinf(r)
        u, du = self.solve(r)
        m = self.amp*r**2*np.exp(u)*du
        if not isinstance(r, float):
            m[mask] = np.inf
        # return scala when the input is scala
        return m.squeeze()[()]

    def total_pressure(self, r):
        u, _ = self.solve(r)
        return 1 - self.amp*u

    @property
    def sigma(self):
        """Compute or return the velocity dispersion"""
        if self._sigma is None:
            self._sigma = self.compute_sigma()
        return self._sigma

    def compute_sigma(self):
        """Calculate mass-weighted mean velocity dispersion.

        Parameters
        ----------
        """
        def _func(r):
            rho = self.density(r)
            sigma2 = self.velocity_dispersion(r)**2
            return r**2*rho*sigma2
        num, _ = quad(_func, 0, self.rcrit, epsrel=1e-6)
        def _func(r):
            rho = self.density(r)
            return r**2*rho
        den, _ = quad(_func, 0, self.rcrit, epsrel=1e-6)
        return np.sqrt(num/den)

    def solve(self, xi):
        """Solve equilibrium equation

        Parameters
        ----------
        xi : array
            Dimensionless radii

        Returns
        -------
        u : array
            Log density u = log(rho/rho_c)
        du : array
            Derivative of u: d(u)/d(xi)
        """
        if isinstance(xi, float) and np.isinf(xi):
            return np.inf, 0
        else:
            mask = np.isinf(xi)

        xi = np.array(xi, dtype='float64')
        y0 = np.array([0, 0])
        if xi.min() > self._rfloor:
            xi = np.insert(xi, 0, self._rfloor)
            istart = 1
        else:
            istart = 0
        if np.all(xi <= self._rfloor):
            u = y0[0]*np.ones(xi.size)
            du = y0[1]*np.ones(xi.size)
        else:
            y = odeint(self._dydx, y0, np.log(xi))
            u = y[istart:, 0]
            du = y[istart:, 1]/xi[istart:]

        if not isinstance(xi, float):
            u[mask] = np.inf
            du[mask] = 0

        # return scala when the input is scala
        return u.squeeze()[()], du.squeeze()[()]

    def critical_radius(self):
        """Find critical TES radius

        Returns
        -------
        float
            Critical radius
        """
        xi = np.logspace(0, 3, 512)
        kappa = self.get_bulk_modulus(xi)
        idx = (kappa < 0).nonzero()[0]
        if len(idx) < 1:
            # kappa is everywhere positive, meaning that this TES
            # is stable at every radius. Return inf.
            return np.inf
        else:
            idx = idx[0] - 1
        func = lambda x: self.get_bulk_modulus(10**x)

        x0, x1 = np.log10(xi[idx]), np.log10(xi[idx+1])
        try:
            logrmax = newton(func, x0, x1=x1).squeeze()[()]
        except:
            logrmax = np.nan
        return 10**logrmax

    def get_bulk_modulus(self, xi):
        # Perturbation with the fixed turbulent velocity field
        # i.e., p = r_s = const.
        u, du = self.solve(xi)
        dpsi = self.amp*np.exp(u)*du
        num = 1 - dpsi**2/18/(1 - self.amp*u)
        den = 1 - dpsi / (9*xi)
        kappa = num / den
        return kappa

    def _dydx(self, y, t):
        """Differential equation for hydrostatic equilibrium.

        Parameters
        ----------
        y : array
            Vector of dependent variables
        x : array
            Independent variable

        Returns
        -------
        array
            Vector of (dy/dx)
        """
        y1, y2 = y
        dy1 = y2

        a = self.amp
        b = 2*self.amp
        c = -9*np.exp(2*t - 2*y1)
        dy2 = -(b/a)*y2 - (c/a)
        return np.array([dy1, dy2])


def minimum_sonic_radius(pindex, rtol=1e-3):
    a = 1e-6
    b = 999

    def func(x):
        tsc = TES(pindex=pindex, rsonic=x)
        return tsc.rcrit

    while ((b - a)/a > rtol):
        c = (a + b)/2
        if np.isinf(func(c)):
            a = c
        else:
            b = c
    return b


def find_sonic_radius(pindex, sigma_target):
    if sigma_target==0:
        return np.infty
    def get_sigma(pindex, rsonic):
        ts = TES(pindex=pindex, rsonic=rsonic)
        return ts.sigma
    rs_min = minimum_sonic_radius(pindex=pindex)
    ts = TES(pindex=pindex, rsonic=rs_min)
    if ts.sigma < sigma_target:
        raise ValueError(f"sigma = {sigma_target:.2f} is too large."
                         " Cannot find the critical radius due to"
                         " the steep dependence of rcrit on rsonic")
    rsonic = brentq(lambda x: get_sigma(pindex, x) - sigma_target, rs_min, 1e5)
    return rsonic


def find_logotrope_amp(sigma_target):
    def _func(A):
        ts = Logotrope(A)
        return ts.sigma - sigma_target
    amp = brentq(_func, 0.15, 1)
    return amp
