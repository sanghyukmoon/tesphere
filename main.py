import tes
from pyathena.core_formation import tools
import numpy as np
import pickle

pindex = [0.3, 0.5, 0.7]
sigma_min, sigma_max = 0.1, 20

for p in pindex:
    velocity_dispersions = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 100)
    rhoc, rhoe, radius, rsonic, mass, f, robs, mobs  = [], [], [], [], [], [], [], []
    for sigma in velocity_dispersions:
        ts = tes.TES('crit', sigma=sigma, p=p)
        rcrit = ts.rmax
        mcrit = ts.menc(rcrit)
        rhoc.append(ts.rho(0))
        rhoe.append(ts.rho(rcrit))
        radius.append(rcrit)
        rsonic.append(ts.rs)
        mass.append(mcrit)
        f.append(ts.f(rcrit))
        fwhm = tools.fwhm(ts.rho, rcrit)
        robs.append(fwhm)
        mobs.append(ts.menc(fwhm))
    rhoc = np.array(rhoc)
    rhoe = np.array(rhoe)
    radius = np.array(radius)
    rsonic = np.array(rsonic)
    mass = np.array(mass)
    f = np.array(f)
    robs = np.array(robs)
    mobs = np.array(mobs)

    res = dict(rhoc=rhoc,
               rhoe=rhoe,
               rcrit=radius,
               mcrit=mass,
               rsonic=rsonic,
               f=f,
               robs=robs,
               mobs=mobs,
               sigma=velocity_dispersions)
    fname = f"data/p{p}.p"
    with open(fname, "wb") as handle:
        pickle.dump(res, handle)
