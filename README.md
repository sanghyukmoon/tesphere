# TES - Turbulent Equilibrium Sphere

A python class `TES` implements the turbulent equilibrium sphere (TES) model presented in Moon & Ostriker (2024, under review).
The TES is a family of equilibrium solutions to the angle-averaged equations of hydrodynamics, with the following assumptions:
* The second-order structure function for the turbulent velocity field is given by a power-law (i.e., there is a power-law linewidth-size relation).
* The turbulence is isotropic.
* Rotation is negligible.

A class `TES` is initialized with the two parameters describing the turbulent velocity field $\left<\delta v_r^2\right>^{1/2}_\rho = \left(\dfrac{r}{r_s}\right)^p$
* `pindex` : the power-law index $p$
* `rsonic` : the dimensionless sonic radius $\xi_s \equiv \frac{\sqrt{4\pi G \rho_c}}{c_s} r_s$

`TES` then calculates the critical quantities which are saved as the following attributes:
* `rcrit` : the dimensionless critical radius $\xi_\mathrm{crit}$
* `mcrit` : the dimensionless critical mass $m_\mathrm{crit}$
* `ucrit` : the logarithmic center-to-edge density contrast $u_\mathrm{crit}$


## Examples


### Radial density profiles

```
from turb_sphere import tes
for rsonic in [np.infty, 5]:
    ts = tes.TES(pindex=0.5, rsonic=rsonic)
    r = np.logspace(-1, np.log10(ts.rcrit))
    plt.loglog(r, ts.density(r), label=r'$\xi_s = {}$'.format(rsonic))
plt.title('Density profiles of critical TESs')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\rho/\rho_c$')
plt.legend()
```
![example](https://github.com/sanghyukmoon/turbulent_equilibrium_sphere/assets/13713687/b3da38b3-4d3e-4174-9ba2-05a74fd1618e)


### Critical quantities
```
from turb_sphere import tes
ts = tes.TES()
print("Bonnor-Ebert sphere")
print(f"xi_crit = {ts.rcrit:.2f}, m_crit = {ts.mcrit:.2f}, "
      f"u_crit = {ts.ucrit:.2f}, sigma = {ts.sigma:.2f}")


ts = tes.TES(pindex=0.5, rsonic=5)
print("Turbulent Equilibrium Sphere")
print(f"xi_crit = {ts.rcrit:.2f}, m_crit = {ts.mcrit:.2f}, "
      f"u_crit = {ts.ucrit:.2f}, sigma = {ts.sigma:.2f}")
```
Output:
```
Bonnor-Ebert sphere
xi_crit = 6.45, m_crit = 15.70, u_crit = 2.64, sigma = 0.00
Turbulent Equilibrium Sphere
xi_crit = 12.95, m_crit = 47.04, u_crit = 3.50, sigma = 1.25
```



