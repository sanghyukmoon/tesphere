# TES - Turbulent Equilibrium Sphere

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


### Radial density profile

```
from turb_sphere import tes
ts = tes.TES(pindex=0.5, rsonic=5)
r = np.logspace(-1, np.log10(ts.rcrit))
plt.loglog(r, ts.density(r))
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\rho/\rho_c$')
```

