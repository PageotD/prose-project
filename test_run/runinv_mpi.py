# Import python modules
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import time
import json
import configparser
import sys
from mpi4py import MPI

from pyprose import PSVModeling
from pyprose import MASW
from pyprose import Swarm
from pyprose.modgen import modelgenerator

def configuration(cfgfile):
    
    # Initialize dictionaries
    elastpar = dict()
    acquiparlist = dict()
    maswpar= dict()
    psopar = dict()
    
    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(cfgfile)

    # Elastpar
    vs = np.load(config['ELASTPAR']['vsmodel'])
    vp = vs * float(config['ELASTPAR']['vpvsratio'])
    rho = np.ones(vs.shape, dtype=np.float32) * float(config['ELASTPAR']['density'] )
    elastpar.update({
        "n1": int(config['ELASTPAR']['n1']),
        "n2": int(config['ELASTPAR']['n2']),
        "dh": float(config['ELASTPAR']['dh']),
        "npml": int(config['ELASTPAR']['npml']),
        "apml": float(config['ELASTPAR']['apml']),
        "ppml": int(config['ELASTPAR']['ppml']),
        "isurf": int(config['ELASTPAR']['isurf']),
        "vpvsratio": float(config['ELASTPAR']['vpvsratio']),
        "vp": vp, "vs": vs, "rho": rho
        })

    # Acquiparlist
    acquiparlist = []
    geoms = config['ACQUIPAR']['geom'].split(",")
    for geom in geoms:
        xr = []
        zr = []
        acqui = np.loadtxt(geom, comments="#")
        for coords in acqui:
            if coords[2] == 1:
                xs = coords[0]
                zs = coords[1]
            else:
                xr.append(coords[0])
                zr.append(coords[1])
        acquiparlist.append({
            "nt": int(config['ACQUIPAR']['nt']),
            "dt": float(config['ACQUIPAR']['dt']),
            "f0": float(config['ACQUIPAR']['f0']),
            "t0": float(config['ACQUIPAR']['t0']),
            "xs": xs, "zs": zs, "xr": np.array(xr, dtype=np.float32), "zr": np.array(zr, dtype=np.float32)
        })

    # MASWPAR
    maswpar.update({
        "fmin": float(config['MASWPAR']['fmin']),
        "fmax": float(config['MASWPAR']['fmax']),
        "vmin": float(config['MASWPAR']['vmin']),
        "vmax": float(config['MASWPAR']['vmax']),
        "deltav": float(config['MASWPAR']['deltav'])
    })

    # PSO
    psopar = dict()
    psopar.update({
        "nparticles": int(config['PSOPAR']['nparticles']),
        "nparameters": int(config['PSOPAR']['nparameters']),
        "niter": int(config['PSOPAR']['niter'])
    })
    return elastpar, acquiparlist, maswpar, psopar

# Specific functions
def multisrc(elastpar, acquiparlist, maswpar):
    # Initialize list of dispersion diagrams
    dispdiags = []
    # Loop over acquisition parameter sets
    for acquipar in acquiparlist:
        # Initialize PSV engine
        modeling = PSVModeling(elastpar, acquipar)
        # Run PSV modeling
        _, wavez = modeling.run()
        # Calculate the MASW diagram
        masw = MASW(wavez, acquipar)
        diag = masw.diagram(maswpar['fmin'], maswpar['fmax'], maswpar['vmin'], maswpar['vmax'], maswpar['deltav'])
        # Normalize
        _, nw = diag.shape
        for iw in range(nw):
            diag[:, iw]/np.amax(diag[:, iw])
        # Add diagram to list
        dispdiags.append(diag)

    return dispdiags

def genmodel(xp, zp, values, n1, n2, elastpar, acquiparlist):
    # Generate a smooth model using set of points

    # Add a random epsilon value to avoid singular matrix
    #xp += np.random.normal(0, 0.0000000001, size=len(xp))
    #zp += np.random.normal(0, 0.0000000001, size=len(zp))
    
    # Create a grid of coordinates for the fine grid
    x_fine = np.linspace(np.min(xp), np.max(xp), n2)
    z_fine = np.linspace(np.min(zp), np.max(zp), n1)
    x_fine_grid, z_fine_grid = np.meshgrid(x_fine, z_fine)

    # Perform RBF interpolation
    epsilon = 1e-10
    try:
        rbf = Rbf(xp, zp, values, function='linear', epsilon=epsilon, smooth=0)
    except Exception:
        # Add a random epsilon value to avoid singular matrix
        xp *= np.random.normal(1.001, 1.02, size=len(xp))
        zp *= np.random.normal(1.001, 1.02, size=len(zp))
        rbf = Rbf(xp, zp, values, function='linear', epsilon=epsilon, smooth=0)

    interpolated_values = rbf(x_fine_grid, z_fine_grid)

    return np.array(interpolated_values, dtype=np.float32)

def initswarm(elastpar, acquiparlist, npts, nparticles):
    # Initialize the SWARM

    # Particles are set to move horizontally between the first and the last receiver
    xmin = np.amin(acquiparlist[0]['xr'])#*elastpar['dh']
    xmax = np.amax(acquiparlist[0]['xr'])#*elastpar['dh']
    # Maximum displacement for particle
    deltax = xmax/2.
    
    # Particles are set to move vertically between 1m and max depth
    zmin = 1.
    zmax = int(elastpar['n1']-1)*elastpar['dh']-1
    # Maximum displacement for particle
    deltaz = zmax/2.

    # Vmin and vmax are estimated from min/max real model values
    vmin = np.amin(elastpar['vs'])*0.8
    vmax = np.amax(elastpar['vs'])*1.2
    deltav = (vmax-vmin)/2.

    # Initialize Swarm
    swarm = Swarm()
    # Initialize parameter space
    swarm.init_pspace(npts=npts, npar=3)
    for ipts in range(npts):
        swarm.pspace[ipts, 0, 0] = xmin
        swarm.pspace[ipts, 0, 1] = xmax
        swarm.pspace[ipts, 0, 2] = deltax
        swarm.pspace[ipts, 1, 0] = zmin
        swarm.pspace[ipts, 1, 1] = zmax
        swarm.pspace[ipts, 1, 2] = deltaz
        swarm.pspace[ipts, 2, 0] = vmin
        swarm.pspace[ipts, 2, 1] = vmax
        swarm.pspace[ipts, 2, 2] = deltav
    
    # Initialize particles
    swarm.init_particles(nparticles, ncvt=0)

    for iparticle in range(nparticles):
        xp = swarm.current[iparticle, :, 0]
        zp = swarm.current[iparticle, :, 1]
        val = swarm.current[iparticle, :, 2]
        model = genmodel(xp, zp, val, elastpar['n1'], elastpar['n2'], elastpar, acquiparlist)
    return swarm

def l2norm(refdiags, tempdiags):
    l2num = 0.
    l2den = 0.
    ndiags = len(refdiags)
    nv, nw = refdiags[0].shape
    n = nv * nw * ndiags
    for idiag in range(ndiags):
        rdiag = refdiags[idiag]
        tdiag = tempdiags[idiag]
        for iw in range(nw):
            for iv in range(nv):
                l2num += (rdiag[iv,iw]-tdiag[iv,iw])**2
                l2den += float(n)*(rdiag[iv,iw])**2
    return np.float32(np.sqrt(l2num/(l2den))*100.)

#============================================================================
# MAIN
#============================================================================

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    elastpar, acquiparlist, maswpar, psopar = configuration(sys.argv[1])
else:
    elastpar = None
    acquiparlist = None
    maswpar = None
    psopar = None
elastpar = comm.bcast(elastpar, root=0)
acquiparlist = comm.bcast(acquiparlist, root=0)
maswpar = comm.bcast(maswpar, root=0)
psopar = comm.bcast(psopar, root=0)

if rank == 0:
    # Processus de référence calcule le modèle de référence
    start = time.time()
    print("calc reference", flush=True)
    refdiags = multisrc(elastpar, acquiparlist, maswpar)
    print("references:: ", time.time()-start, flush=True)
else:
    # Creation de la variable sur les autres processus
    refdiags = None

# Diffusion du modèle de référence à tous les processus
refdiags = comm.bcast(refdiags, root=0)

# Générer les modèles de départ avec PSO
niter = psopar['niter']
npts = psopar['nparameters']
nparticles = psopar['nparticles']

if rank == 0:
    # Initialisation de l'essaim
    swarm = initswarm(elastpar,  acquiparlist, npts, nparticles)
    swarm.misfit[:] = 0.

current = np.zeros((nparticles, npts, 3), dtype=np.float32)

for iter in range(niter):
    # Diffusion du modèle de référence à tous les processus
    if rank == 0:
        print(f"ITER:: {iter+1}/{niter}", flush=True)
        current = swarm.current.copy()
        #print("ORIGINAL CURRENT", rank, current)
        #print(current)
    else:
        current = np.zeros((nparticles, npts, 3), dtype=np.float32)
    current = comm.bcast(current, root=0)
    comm.Barrier()
    # boucle sur les particules/modèles
    start = time.time()
    misfit = np.zeros(nparticles, dtype=np.float64)
    #misfit = [0.] * nparticles 
    for iparticle in range(rank, nparticles, size):
        # générer les modeles
        print(f"particle :: {iparticle+1}/{nparticles}...rank :: {rank}", flush=True)
        xp = current[iparticle, :, 0]
        zp = current[iparticle, :, 1]
        val = current[iparticle, :, 2]
        vs = genmodel(xp, zp, val, elastpar['n1'], elastpar['n2'], elastpar, acquiparlist)
        vp = vs * elastpar['vpvsratio']
        elastpar['vp'] = vp
        elastpar['vs'] = vs        
        # Calcul des diagrammes
        tempdiags = multisrc(elastpar, acquiparlist, maswpar)
        # Calcul du misfit
        misfit[iparticle] = l2norm(refdiags, tempdiags)
        
    comm.Barrier()
    # Collecter les misfits de tous les processus
    reduce_misfit = comm.reduce(misfit, root=0)

    print(f"Rank: {rank}...Exectime {time.time()-start}", flush=True)

    # Processus principal effectuant la mise à jour de l'essaim
    if rank == 0:
        #print(all_misfit)
        all_misfit = reduce_misfit #np.array(reduce_misfit, dtype=np.float32)
        datad = []
        for iparticle in range(nparticles):

            print(f"particle :: {iparticle+1}... L2: {round(all_misfit[iparticle],5)}|{round(swarm.misfit[iparticle],5)}...", flush=True)

            if iter == 0 or round(all_misfit[iparticle],5) <= round(swarm.misfit[iparticle],5):
                swarm.misfit[iparticle] = round(all_misfit[iparticle],5)
                swarm.history[iparticle, :, :] = swarm.current[iparticle, :, :]
            
            xp = swarm.history[iparticle, :, 0]
            zp = swarm.history[iparticle, :, 1]
            val = swarm.history[iparticle, :, 2]
            misfit = all_misfit[iparticle]
            datad.append({
                'particule': iparticle, 'xp': xp.tolist(), 'zp': zp.tolist(), 'vs': val.tolist(), 'misfit': misfit.item()
            })

        with open(f"data/iter_{str(iter+1).zfill(3)}.json", "w") as f:
            f.write(json.dumps(datad))

        # Get the best particle
        ibest = np.argmin(swarm.misfit)
        print(f"Best:{np.amin(swarm.misfit)}... Mean: {np.mean(swarm.misfit)}")

        #swarm.update(topology='full')
        swarm.bbupdate(topology='full')