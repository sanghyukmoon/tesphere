from scipy.integrate import solve_ivp, quad, odeint
from scipy.optimize import minimize_scalar, brentq, newton
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


class TESc:
    """Turbulent equilibrium sphere of a fixed central density.

    Parameters
    ----------
    p : float, optional
        power-law index of the velocity dispersion.
    rs : float, optional
        dimensionless sonic radius.

    Attributes
    ----------
    p : float
        power-law index of the velocity dispersion.
    rs : float
        dimensionless sonic radius.
    """
    def __init__(self, p=0.5, rs=np.inf, sigma=0, compute_rcrit=True):
        self._rfloor = 1e-5
        self._rs_ceil = 1e5
        self.p = p
        if sigma > 0:
            def get_sigv(rs, p):
                tsc = TESc(p=p, rs=rs, compute_rcrit=False)
                sigv = tsc.sigma()
                return sigv
            self._rs_floor = self.sonic_radius_floor()
            tsc = TESc(p=p, rs=self._rs_floor, compute_rcrit=False)
            if tsc.sigma() < sigma:
                raise ValueError(f"sigma = {sigma:.2f} is too large. Cannot find xi_crit due to"
                                 " the steep dependence of xi_crit on xi_s")
            self.rs = brentq(lambda x: get_sigv(x, p) - sigma,
                             self._rs_floor, self._rs_ceil)
        else:
            self.rs = rs
        if compute_rcrit:
            self.rmax = self.critical_radius()

    def rho(self, r):
        """Calculate normalized density.

        Notes
        -----
        rho = rho_c * e^u
        """
        u, _ = self.solve(r)
        return np.exp(-u)

    def critical_radius(self):
        """Find critical TES radius

        Returns
        -------
        float
            Critical radius
        """
        xi = np.logspace(0, 2, 512)
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

    def menc(self, xi0):
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
        f = self.f(xi0)
        df = 2*self.p*(f - 1)/xi0
        m = xi0**2 / np.sqrt(4*np.pi) * (f*du - df)
        if not isinstance(xi0, float):
            m[mask] = np.inf
        # return scala when the input is scala
        return m.squeeze()[()]

    def get_bulk_modulus(self, xi):
        # Perturbation with the fixed turbulent velocity field
        # i.e., p = r_s = const.
        u, du = self.solve(xi)
        m = self.menc(xi)
        f = self.f(xi)
        dsu, dslogm = self._get_sonic_radius_derivatives(xi)
        dslogf = -2*self.p*(1 - 1/f)
        num = 1 - 0.5*dsu + 0.5*dslogf\
                - 2*np.pi*m**2/(f*np.exp(-u)*xi**4)*(1 - dslogm)
        denom = 1 - np.sqrt(4*np.pi)*m/(xi**3*np.exp(-u))*(1 - dslogm)
        kappa = (2/3)*num/denom
        return kappa

    def dv(self, r):
        return (r / self.rs)**(self.p)

    def f(self, r):
        return 1 + self.dv(r)**2

    def sigma(self):
        """Calculate mass-weighted mean velocity dispersion within rcrit

        Returns
        -------
        sigv : float
            Mass-weighted radial velocity dispersion
        """
        rc = self.critical_radius()
        def func(xi):
            u, _ = self.solve(xi)
            dm = xi**2*np.exp(-u)
            dv2 = self.dv(xi)**2
            return dm*dv2
        num, _ = quad(func, self._rfloor, rc)

        def func(xi):
            u, _ = self.solve(xi)
            dm = xi**2*np.exp(-u)
            return dm
        den, _ = quad(func, self._rfloor, rc)

        sigv = np.sqrt(num/den)
        return sigv

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
            y = odeint(self._dydx, y0, np.log(xi),
                       Dfun=self._jac, col_deriv=True)
            u = y[istart:, 0]
            du = y[istart:, 1]/xi[istart:]

        if not isinstance(xi, float):
            u[mask] = np.inf
            du[mask] = 0

        # return scala when the input is scala
        return u.squeeze()[()], du.squeeze()[()]

    def sonic_radius_floor(self, rtol=1e-3):
        a = 1e-6
        b = 999

        def func(x):
            tsc = TESc(p=self.p, rs=x)
            return tsc.rmax

        while ((b - a)/a > rtol):
            c = (a + b)/2
            if np.isinf(func(c)):
                a = c
            else:
                b = c
        return b

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

        f = self.f(x)
        df = 2*self.p*(f - 1)
        ddf = 2*self.p*df

        a = f
        b = f + df
        c = -df - ddf - np.exp(2*t - y1)
        dy2 = -(b/a)*y2 - (c/a)
        return np.array([dy1, dy2])

    def _jac(self, y, t):
        y1, y2 = y
        x = np.exp(t)
        f = self.f(x)
        df = 2*self.p*(f - 1)
        j11 = 0
        j12 = 1
        j21 = -(1./f)*np.exp(2*t - y1)
        j22 = -(1 + df/f)
        return [[j11, j21], [j12, j22]]

    def _get_sonic_radius_derivatives(self, xi, dlog_xi_s=1e-3):
        log_xi_s = np.log(self.rs)

        rs = [np.exp(log_xi_s - dlog_xi_s),
                np.exp(log_xi_s + dlog_xi_s)]
        tsc = [TESc(p=self.p, rs=rs[0], compute_rcrit=False),
               TESc(p=self.p, rs=rs[1], compute_rcrit=False)]
        u = [tsc[0].solve(xi)[0], tsc[1].solve(xi)[0]]
        m = [tsc[0].menc(xi), tsc[1].menc(xi)]
        du = (u[1] - u[0]) / (2*dlog_xi_s)
        dlogm = (np.log(m[1]) - np.log(m[0])) / (2*dlog_xi_s)

        return du, dlogm
