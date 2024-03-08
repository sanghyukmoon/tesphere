from turb_sphere import tes, utils
import numpy as np
import pickle
from pathlib import Path
import sys


def run_TES():
    sigma_min, sigma_max = 0.05, 20

    pindex_list = np.array(sys.argv[1:], dtype=float)
    for p in pindex_list:
        print(f"Generate data for p = {p}")
        velocity_dispersions = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 1024)
        velocity_dispersions = np.insert(velocity_dispersions, 0, 0)
        rhoc, rhoe, radius, mass = [], [], [], []
        rsonic, f, robs, mobs, sigma_obs = [], [], [], [], []
        for sigma in velocity_dispersions:
            ts = tes.TES('crit', sigma=sigma, p=p, from_table=False)
            rcrit = ts.rmax
            mcrit = ts.menc(rcrit)
            rhoc.append(ts.rho(0))
            rhoe.append(ts.rho(rcrit))
            radius.append(rcrit)
            rsonic.append(ts.rs)
            mass.append(mcrit)
            f.append(ts.f(rcrit))

            # Observable properties
            fwhm = utils.fwhm(ts.rho, rcrit)
            robs.append(fwhm)
            mobs.append(ts.menc(fwhm))
            num = utils.integrate_2d_projected(lambda x: ts.rho(x)*ts.vr(x)**2, fwhm, ts.rmax)
            den = utils.integrate_2d_projected(lambda x: ts.rho(x), fwhm, ts.rmax)
            sigma_obs.append(np.sqrt(num/den))
        rhoc = np.array(rhoc)
        rhoe = np.array(rhoe)
        radius = np.array(radius)
        rsonic = np.array(rsonic)
        mass = np.array(mass)
        f = np.array(f)
        robs = np.array(robs)
        mobs = np.array(mobs)
        sigma_obs = np.array(sigma_obs)

        res = dict(rhoc=rhoc,
                   rhoe=rhoe,
                   rcrit=radius,
                   mcrit=mass,
                   rsonic=rsonic,
                   f=f,
                   robs=robs,
                   mobs=mobs,
                   sigma=velocity_dispersions,
                   sigma_obs=sigma_obs)
        fp = Path(__file__).parent / f"data/p{p}.p"
        with open(fp, "wb") as handle:
            pickle.dump(res, handle)


def run_TESc():
    pindex_list = np.array(sys.argv[1:], dtype=float)
    for p in pindex_list:
        print(f"Generate data for p = {p}")

        # Set minimum and maxmimum velocity dispersion
        ts = tes.TESc(p=p, rs=1e2)
        sigma_min = ts.sigma
        rfloor = ts.sonic_radius_floor()
        ts = tes.TESc(p=p, rs=rfloor)
        sigma_max = ts.sigma - 1e-6  # avoid exceeding sigma_max by truncation error in log10

        velocity_dispersions = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 1024)
        velocity_dispersions = np.insert(velocity_dispersions, 0, 0)
        ucrit, rcrit, mcrit, rsonic = [], [], [], []
        for sigma in velocity_dispersions:
            ts = tes.TESc(sigma=sigma, p=p)
            ucrit.append(np.log(1/ts.rho(ts.rmax)))
            rcrit.append(ts.rmax)
            mcrit.append(ts.menc(ts.rmax))
            rsonic.append(ts.rs)
        ucrit = np.array(ucrit)
        rcrit = np.array(rcrit)
        mcrit = np.array(mcrit)
        rsonic = np.array(rsonic)

        res = dict(ucrit=ucrit,
                   rcrit=rcrit,
                   mcrit=mcrit,
                   rsonic=rsonic)
        fp = Path(__file__).parent / f"data/tsc.p{p}.p"
        with open(fp, "wb") as handle:
            pickle.dump(res, handle)

if __name__ == "__main__":
    run_TESc()
