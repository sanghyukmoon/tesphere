from scipy.integrate import odeint
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
import numpy as np
import logging


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
        self.p = p
        self.rs = rs

        if uc == 'crit':
            self.uc = self.find_ucrit()
        elif isinstance(uc, (float, int)):
            self.uc = uc
        else:
            raise ValueError("uc must be either float or 'crit'")

        # Do root finding in logarithmic space to find rmax
        def _calc_u(logr):
            r = np.exp(logr)
            u, _ = self.solve(r)
            return u
        logrmax = brentq(_calc_u, np.log(self._rfloor), np.log10(self._rceil))
        self.rmax = np.exp(logrmax)

    def rho(self, r):
        """Calculate normalized density.

        Notes
        -----
        rho = (P_ext / c_s^2) * (e^u / f)
        """
        u, _ = self.solve(r)
        return np.exp(u) / self.f(r)

    def f(self, r):
        """Ratio of total to thermal pressure.

        Notes
        -----
        This dimensionless function is defined as
            f = 1 + (r / r_s)^(2*p)
        such that Peff = f*Pthm.
        """
        return 1 + (r / self.rs)**(2*self.p)

    def menc(self, r):
        """Dimensionless enclosed mass

        Notes
        -----
        Menc = c_s^4 * G^(-3/2) * P_ext^(-1/2) * menc
        """
        _, du = self.solve(r)
        f = self.f(r)
        return -f*r**2*du/np.sqrt(4*np.pi)

    def find_ucrit(self):
        """Find the critical uc

        Returns
        -------
        ucrit : float
            The critical value of uc.

        Notes
        -----
        The root is not necessarily be bracketed by `bracket`.
        The `bracket` argument is used as initial points
        for a downhill bracket search (see `scipy.optimize.bracket`)
        Here, bracket is set to enclose the root for BE sphere.
        For more turbulent TES, the end of the bracket will be
        extended by the downhill bracket search.
        """
        def _func(uc):
            ts = TES(uc, p=self.p, rs=self.rs)
            return -ts.menc(ts.rmax)
        res = minimize_scalar(_func, bracket=(2.64, 2.65))
        ucrit = res.x
        return ucrit

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
