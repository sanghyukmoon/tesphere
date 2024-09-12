[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13661169.svg)](https://doi.org/10.5281/zenodo.13661169)

# tesphere - Turbulent Equilibrium Sphere

`tesphere` is a python package that implements the turbulent equilibrium sphere (TES) model introduced in Moon & Ostriker (2024, in press).
The TES is a family of equilibrium solutions to the angle-averaged equations of hydrodynamics under the following assumptions:
* The second-order structure function for the turbulent velocity field is given by a power-law (i.e., there is a power-law linewidth-size relation).
* The turbulence is isotropic.
* Rotation is negligible.

A class `tesphere.TES` is initialized with the two parameters describing the turbulent velocity field $\left<\delta v_r^2\right>^{1/2}_\rho = \left(\dfrac{r}{r_s}\right)^p$
* `pindex` : the power-law index $p$
* `rsonic` : the dimensionless sonic radius $\xi_s \equiv \frac{\sqrt{4\pi G \rho_c}}{c_s} r_s$
  
which then determine the following attributes.
* `rcrit` : the dimensionless critical radius $\xi_\mathrm{crit}$
* `mcrit` : the dimensionless critical mass $m_\mathrm{crit}$
* `ucrit` : the logarithmic center-to-edge density contrast $u_\mathrm{crit}$

# Author

Sanghyuk Moon (sanghyuk.moon at princeton dot edu)

# Examples

An example *jupyter notebook* file that reproduces the figures in Moon & Ostriker (2024) can be found in `example` folder. A few simple examples are provided below as a preview.

## Volume density profiles

```
from tesphere import tes
r = np.logspace(-1, 2)
lines, labels = [], []
linestyles = ['-', '--', '-.']
for ls, rs in zip(linestyles, [np.infty, 7, 3]):
    ts = tes.TES(rsonic = rs)
    ln, = plt.loglog(r, ts.density(r), c='k', ls=ls, lw=1.5)
    lines.append(ln)
    labels.append(r'$\xi_s = {}$'.format(rs))
labels[0] = r'$\xi_s = \infty$'
plt.xlim(1e-1, 1e2)
plt.ylim(1e-2, 2e0)
plt.legend(lines, labels, loc='upper right')
plt.xlabel(r'$\xi = \dfrac{(4\pi G \rho_c)^{1/2}}{c_s}r$')
plt.ylabel(r'$e^{-u} = \left<\rho\right>/\rho_c$')
```
![example1](https://github.com/user-attachments/assets/82017b47-8fd2-4b85-b9fd-5844d6cd3c5b)

## Column density profiles of critical TESs
```
from tesphere import tes, utils
pindex = 0.5
for sigma, ls, c in zip([0, 1, 2], ['-', '--', ':'], ['tab:brown', 'tab:olive', 'tab:cyan']):
    rs = tes.TES.find_sonic_radius(pindex, sigma)
    ts = tes.TES(pindex=pindex, rsonic=rs)
    rcyl = np.linspace(0, ts.rcrit, 512)
    robs = utils.fwhm(ts.density, ts.rcrit)
    dcol = utils.integrate_los(ts.density, rcyl, ts.rcrit)
    R0 = 4*np.pi/dcol[0]
    plt.plot(rcyl/R0, dcol/dcol[0], c=c, label=r"$\sigma_\mathrm{1D}/c_s = $"+f"${sigma}$", ls=ls)
    dcol_fwhm = utils.integrate_los(ts.density, robs, ts.rcrit)
    plt.plot(robs/R0, dcol_fwhm/dcol[0], marker=MarkerStyle('^', fillstyle='full'), c=c, ms=12)

plt.legend()
plt.xlim(0, 11)
plt.ylim(0, 1)
plt.xlabel(r'$R/(c_s^2G^{-1}\Sigma_c^{-1})$')
plt.ylabel(r'$\Sigma/\Sigma_c$')
```
![example2](https://github.com/user-attachments/assets/9c593837-7bd2-4509-abff-a3990634b92c)


