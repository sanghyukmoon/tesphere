from scipy.integrate import odeint, simpson, quad
from scipy.optimize import minimize_scalar, brentq, newton
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging
from pyathena.core_formation import tools


class TES:
    """Turbulent equilibrium sphere at a fixed external pressure.

    External pressure includes the thermal and turbulent pressures.

    Parameters
    ----------
    uc : float
        Logarithm of the center-to-edge pressure contrast.
    p : float, optional
        Power-law index of the linewidth-size relation.
    rs : float, optional
        Dimensionless sonic radius.

    Attributes
    ----------
    uc : float
        Logarithm of the center-to-edge pressure contrast.
    p : float
        Power-law index of the linewidth-size relation.
    rs : float
        Dimensionless sonic radius.
    rmax : float
        Maximum radius.
    _rfloor : float
        Minimum radius.
    _rceil : float
        Maximum radius.

    Notes
    -----
    The angle-averaged hydrostatic equation is solved using the following
    dimensionless variables,
        xi = r / r_0,
        u = ln(P_eff/P_ext),
    where
        r_0 = c_s^2 / (4 pi G P_ext)^(1/2)
    is the scale radius and P_ext is the total (thermal + turbulent)
    pressure at the edge.

    The density is given by
        rho = (P_ext / c_s^2) * (exp(u) / f)

    Examples
    --------
    >>> import tes
    >>> ts = tes.TES(1.0)
    >>> r = np.linspace(0, ts.rmax)
    >>> u, du = ts.solve(r, u_crit)
    >>> # plot density profile
    >>> plt.loglog(r, np.exp(u))
    """
    def __init__(self, uc, p=0.5, rs=np.inf):
        self._rfloor = 1e-6
        self._rceil = 1e20
        self.uc = uc
        self.p = p
        self.rs = rs

        # Do root finding in logarithmic space to find rmax
        def _calc_u(logr):
            r = np.exp(logr)
            u, _ = self.solve(r)
            return u
        logrmax = brentq(_calc_u, np.log(self._rfloor), np.log10(self._rceil))
        self.rmax = np.exp(logrmax)

    def rho(self, r):
        u, _ = self.solve(r)
        return np.exp(u) / self.f(r)

    def f(self, r):
        return 1 + (r / self.rs)**(2*self.p)

    def solve(self, rin):
        """Solve equilibrium equation.

        Parameters
        ----------
        rin : float or array
            Dimensionless radii

        Returns
        -------
        u : float or array
        du : float array
            Derivative of u: du/dr
        """
        def _dydx(y, s):
            """ODE in logr space"""
            u, du = y
            f = self.f(np.exp(s))
            # Coefficients of the 2nd order ODE.
            a = f
            b = (2*self.p + 1)*f - 2*self.p
            c = np.exp(u + 2*s)/f
            ddu = -(b/a)*du - (c/a)
            return np.array([du, ddu])

        y0 = np.array([self.uc, 0])  # Initial conditions

        rin = np.array(rin, dtype='float64')
        if np.all(rin <= self._rfloor):
            u = y0[0]*np.ones(rin.size)
            du = y0[1]*np.ones(rin.size)
        else:
            u = np.zeros(rin.size)
            du = np.zeros(rin.size)

            # Use only the part above the floor for ODE integration.
            mask = rin > self._rfloor
            u[~mask] = y0[0]  # Set to IC for r < rfloor
            du[~mask] = y0[1]  # Set to IC for r < rfloor
            r = rin[mask]
            r = np.insert(r, 0, self._rfloor)  # Insert initial value point
            s = np.log(r)

            # Solve ODE
            y = odeint(_dydx, y0, s)
            u[mask] = y[1:, 0]
            du[mask] = y[1:, 1]/r[1:]

        return u.squeeze()[()], du.squeeze()[()]
