from scipy.integrate import solve_ivp, quad, odeint
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle
from pathlib import Path


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
        r_0 = c_s^2 / G^(1/2) / P_ext^(1/2)
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
    def __init__(self, uc, p=0.5, rs=np.inf, sigma=0, from_table=True):
        self._rfloor = 1e-6
        self._rceil = 1e20
        self._rs_max = 1e4
        self._sigma = None
        self.p = p

        if np.isfinite(rs) and sigma > 0:
            raise ValueError("Provide either rs or sigma, but not both.")

        # Set sonic radius based on either rs or sigma
        if sigma > 0:
            if from_table:
                fp = Path(__file__).parent.parent / f"data/p{p}.p"
                ds = pickle.load(open(fp, "rb"))
                self.rs = interp1d(ds['sigma'], ds['rsonic'])(sigma)[()]
            else:
                # Do root finding in log space
                def _func(logrs):
                    ts = TES(uc, p=p, rs=10**logrs)
                    return ts.sigma - sigma
                if _func(np.log10(self._rs_max)) > 0:
                    raise ValueError("The given value of `sigma` is too small.")
                self.set_sonic_radius_floor()
                logrs = brentq(_func, np.log10(self._rs_floor), np.log10(self._rs_max))
                self.rs = 10**(logrs)
        else:
            self.rs = rs

        # Set pressure contrast based on uc
        # The order is important!!! this must come after setting `rs`
        if uc == 'crit':
            # This raises ValueError when rs is too small
            self.uc = self.find_ucrit()
        elif isinstance(uc, (float, int)):
            self.uc = uc
        else:
            raise ValueError("uc must be either float or 'crit'.")

        # Do root finding in logarithmic space to find rmax
        # The order is important!!! this must come after setting `rs` and `uc`
        def _calc_u(logr):
            r = 10**logr
            u, _ = self.solve(r)
            return u
        logrmax = brentq(_calc_u, np.log10(self._rfloor), np.log10(self._rceil))
        self.rmax = 10**logrmax

        # Sanity check
        if sigma > 0:
            if not np.isclose(self.sigma, sigma, atol=1e-2):
                raise ValueError(f"The computed velocity dispersion {self.sigma:.2f} is different"
                                 f" from the input velocity dispersion {sigma:.2f}."
                                 " The interpolation table may be either currupted"
                                 " or has too low resolution.")


    def set_sonic_radius_floor(self, atol=1e-3):
        """Find minimum rs that does not cause ValueError on ODE solve"""
        a = -5
        b = 5
        while (b - a > atol):
            c = (a + b)/2
            try:
                ts = TES('crit', p=self.p, rs=10**c)
            except ValueError:
                a = c
            else:
                b = c
        self._rs_floor = 10**b

    def rho(self, r):
        """Calculate normalized density.

        Notes
        -----
        rho = (P_ext / c_s^2) * (e^u / f)
        """
        u, _ = self.solve(r)
        return np.exp(u) / self.f(r)

    def vr(self, r):
        """Calculate turbulent velocity.

        Notes
        -----
        vr = c_s * (r / r_s)^p
        """
        return (r / self.rs)**self.p

    def density_contrast(self):
        """Calculate center-to-edge density contrast"""
        return np.exp(self.uc) * self.f(self.rmax)

    @property
    def sigma(self):
        """Compute or return the velocity dispersion"""
        if self._sigma is None:
            self._sigma = self.compute_sigma()
        return self._sigma

    def compute_sigma(self, rmax=None):
        """Calculate mass-weighted mean velocity dispersion.

        Parameters
        ----------
        rmax : float, optional
            Outer radius within which the velocity dispersion is computed.
            If not given, use the outer radius.
        """
        if rmax is None:
            rmax = self.rmax
        def _func(r):
            return self.rho(r)*(self.f(r) - 1)*r**2
        num, _ = quad(_func, 0, rmax, epsrel=1e-6)
        def _func(r):
            return self.rho(r)*r**2
        den, _ = quad(_func, 0, rmax, epsrel=1e-6)
        return np.sqrt(num/den)

    def f(self, r):
        """Ratio of total to thermal pressure.

        Notes
        -----
        This dimensionless function is defined as
            f = 1 + (r / r_s)^(2*p)
        such that Peff = f*Pthm.
        It is useful to note that
            sigma^2 / c_s^2 = f - 1
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
        return -f*r**2*du

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
        res = minimize_scalar(_func, bracket=(0, 2.65))
        ucrit = res.x
        return ucrit

#    def solve(self, rin):
#        """Solve equilibrium equation.
#
#        Parameters
#        ----------
#        rin : float or array
#            Dimensionless radii
#
#        Returns
#        -------
#        u : float or array
#        du : float array
#            Derivative of u: du/dr
#        """
#        def _dydx(s, y):
#            """ODE in logr space"""
#            u, du = y
#            f = self.f(np.exp(s))
#            # Coefficients of the 2nd order ODE.
#            a = f
#            b = (2*self.p + 1)*f - 2*self.p
#            c = 4*np.pi*np.exp(u + 2*s)/f
#            ddu = -(b/a)*du - (c/a)
#            return np.array([du, ddu])
#
#        y0 = np.array([self.uc, 0])  # Initial conditions
#
#        rin = np.array(rin, dtype='float64')
#        if np.all(rin <= self._rfloor):
#            u = y0[0]*np.ones(rin.size)
#            du = y0[1]*np.ones(rin.size)
#        else:
#            u = np.zeros(rin.size)
#            du = np.zeros(rin.size)
#
#            # Use only the part above the floor for ODE integration.
#            mask = rin > self._rfloor
#            u[~mask] = y0[0]  # Set to IC for r < rfloor
#            du[~mask] = y0[1]  # Set to IC for r < rfloor
#            r = rin[mask]
#            r = np.insert(r, 0, self._rfloor)  # Insert initial value point
#            s = np.log(r)
#
#            # Solve ODE
#            res = solve_ivp(_dydx, (s[0], s[-1]), y0, t_eval=s, method='Radau')
#            # TODO(SMOON) resolve this
#            if not res.success:
#                print(res)
#                raise ValueError("Cannot solve the ODE")
#            y = res.y
#            u[mask] = y[0, 1:]
#            du[mask] = y[1, 1:]/r[1:]
#        return u.squeeze()[()], du.squeeze()[()]


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
            c = 4*np.pi*np.exp(u + 2*s)/f
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
