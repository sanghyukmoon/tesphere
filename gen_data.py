from turb_sphere import tes, utils
import numpy as np
import pickle
from pathlib import Path
import sys


def run_TES():
    pindex_list = np.array(sys.argv[1:], dtype=float)
    for p in pindex_list:
        print(f"Generate data for p = {p}")

        # Set minimum and maxmimum velocity dispersion
        ts = tes.TES(p=p, rs=1e2)
        sigma_min = ts.sigma
        rfloor = ts.sonic_radius_floor()
        ts = tes.TES(p=p, rs=rfloor)
        sigma_max = ts.sigma - 1e-6  # avoid exceeding sigma_max by truncation error in log10

        velocity_dispersions = np.logspace(np.log10(sigma_min), np.log10(sigma_max), 1024)
        velocity_dispersions = np.insert(velocity_dispersions, 0, 0)
        ucrit, rcrit, mcrit, rsonic = [], [], [], []
        for sigma in velocity_dispersions:
            ts = tes.TES(sigma=sigma, p=p)
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
                   rsonic=rsonic,
                   veldisp=velocity_dispersions)
        fp = Path(__file__).parent / f"data/tsc.p{p}.p"
        with open(fp, "wb") as handle:
            pickle.dump(res, handle)

if __name__ == "__main__":
    run_TES()
