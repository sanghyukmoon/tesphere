# TES - Turbulent Equilibrium Sphere

A class `TES` implements the turbulent equilibrium sphere model developed in Moon & Ostriker (2024).

`TES` takes two parameters, `pindex` and `rsonic`, which correspond to $p$ and $\xi_s$ in Moon & Ostriker (2024), respectively.

The name of the attributes and the corresponding physical quantities (in the notation of Moon & Ostriker (2024)) are listed below:

* `pindex` : $p$
* `rsonic` : $\xi_s$
* `rcrit` : $\xi_\mathrm{crit}$
* `mcrit` : $m_\mathrm{crit}$
* `ucrit` : $u_\mathrm{crit}$


## Examples

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


### Radial density profile

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


