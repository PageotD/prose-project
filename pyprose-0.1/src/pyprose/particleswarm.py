#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: swarm.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018, 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Class and methods for particle swarm optimization.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

class Swarm:
    """
    Class for the paticle swarm optimization (PSO) method.
    *Standard* *barebone* and *fully-informed* declinaison of PSO are available
    with the *inertia weight* and the *constriction factor* strategies. Several
    topologies are lso implemented: *full*, *ring* and *toroidal*.
    """

    def __init__(self):
        """
        Initialize the Swarm class.
        """
        self.current = np.zeros((1, 1, 3), dtype=np.float32)
        self.velocity = np.zeros((1, 1, 3), dtype=np.float32)
        self.history = np.zeros((1, 1, 3), dtype=np.float32)
        self.misfit = np.zeros(1, dtype=np.float32)
        self.current_misfit = np.zeros(1, dtype=np.float32)
        self.pspace = np.zeros((1, 1, 3), dtype=np.float32)
        self.stagnation = np.zeros(1, dtype=np.float32)

    def init_pspace(self, **options):
        """
        Initialiaze parameter space from file

        :param npts: number of independant points
        :param npar: number of parameter per points
        :param fmod: input file containing the boundaries of the parameter space.

        If `fmod` exists, ``npts`` and ``npar`` are deduced from file. Else, an empty
        parameter space is intialize (with ``npts`` and ``npar`` provided by the
        user).

        .. rubric:: Basic usage

        >>> #Import PSO method from NeSSI
        >>> from nessi.globopt import Swarm
        >>> # Create a Swarm object
        >>> swarm = Swarm()
        >>> # Initialize an empty pspace using `npts` and `npar`
        >>> swarm.init_pspace(npts=1, npar=2)
        >>> swarm.pspace
        array([[[0., 0., 0.],
                [0., 0., 0.]]], dtype=float32)

        """

        # Get parameters
        npts = options.get('npts', 1)
        npar = options.get('npar', 1)
        fmod = options.get('fmod', ' ')

        if fmod == ' ':
            # Initialize an empty pspace array of size (npts, npar)
            self.pspace.resize(npts, npar, 3)
        else:
            # Initialize the pspace array from file
            # Load pspace file in a temporary array
            tmp = np.loadtxt(fmod, ndmin=2, comments='#')

            # Check the number of points per particule
            npts = tmp.shape[0]
            npar = int(tmp.shape[1]/3)

            # Resize pspace array
            self.pspace.resize(npts, npar, 3)

            # Fill pspace array
            i = 0
            for ipar in range(0, npar):
                self.pspace[:, ipar, :] = tmp[:, i:i+3]
                i += 3

    def init_particles(self, nindv, ncvt=0):
        """
        Initialize all the particles of the swarm at random position in the
        parameter space.

        :param nindv: integer, number of particles
        :param ncvt: integer, number of iteration for centroidal Voronoi
            tessellation (McQueen algorithm)

        .. rubric:: Basic usage

        >>> # Import the Swarm module
        >>> from nessi.globopt import Swarm
        >>> # Initialize the swarm
        >>> swarm = Swarm()
        >>> # Initialize an empty parameter space
        >>> swarm.init_pspace(npts=1, npar=1)
        >>> # Edit the parameter space
        >>> # The unknown is set to be between -1 and 1 with a maximum velocity
        >>> # of 0.2
        >>> swarm.pspace[0, 0, 0] = -1.
        >>> swarm.pspace[0, 0, 1] = 1.
        >>> swarm.pspace[0, 0, 2] = 0.2
        >>> # Initialize population with 10 particles
        >>> swarm.init_particles(10)
        >>> swarm.current
        array([[[ 0.9879148 ]],
               [[-0.2595402 ]],
               [[-0.85254925]],
               [[ 0.9707204 ]],
               [[-0.5316951 ]],
               [[-0.4493006 ]],
               [[-0.638056  ]],
               [[ 0.4990087 ]],
               [[ 0.9022976 ]],
               [[-0.2530524 ]]], dtype=float32)

        """

        # Get the number of points and the number of parameters from pspace
        npts = self.pspace.shape[0]
        npar = self.pspace.shape[1]

        # Resize arrays
        self.current.resize(nindv, npts, npar)
        self.velocity.resize(nindv, npts, npar)
        self.history.resize(nindv, npts, npar)
        self.misfit.resize(nindv)
        self.current_misfit.resize(nindv)
        self.stagnation.resize(nindv)

        # Initialize arrays
        self.current[:, :, :] = 0.
        self.velocity[:, :, :] = 0.
        self.history[:, :, :] = 0.
        self.misfit[:] = 0.
        self.current_misfit[:] = 0.
        self.stagnation[:] = 0

        # Random generation of particle position
        for indv in range(0, nindv):
            for ipts in range(npts):
                for ipar in range(npar):
                    p_random = np.random.uniform(0., 1.)
                    self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 0] + p_random*(self.pspace[ipts, ipar, 1]-self.pspace[ipts, ipar, 0])
            #print(self.current[indv, :, :])

        # Centroidal Voronoi tessellation
        if ncvt > 0:
            # Initialize
            j = np.zeros(nindv, dtype=np.float32)
            j[:] = 1.
            # Create temporary particle array
            qtmp = np.zeros((npts, npar), dtype=np.float32)
            # Loop over iterations
            for it in range(0, ncvt):
                # Random individual
                p_random = np.random.random_sample((npts, npar))
                qtmp[:, :] = self.pspace[:, :, 0]\
                    + p_random*(self.pspace[:, :, 1]-self.pspace[:, :, 0])
                # Calculate distance
                d = np.zeros(nindv, dtype=np.float32)
                for indv in range(0, nindv):
                    for ipts in range(0,npts):
                        for ipar in range(0, npar):
                            if(self.pspace[ipts, ipar, 2] > 0.):
                                d[indv] += ((self.current[indv,ipts,ipar]-qtmp[ipts,ipar])/self.pspace[ipts,ipar,1])**2
                d[:] = np.sqrt(d[:])
                # Search closest individual
                iclose = np.argmin(d)
                # Correct position
                for ipts in range(0, npts):
                    for ipar in range(0, npar):
                        self.current[iclose,ipts,ipar] = (j[iclose]*self.current[iclose,ipts,ipar]+qtmp[ipts,ipar])/(j[iclose]+1.)
                j[iclose] += 1.

    def _get_neighbors(self, topology, indv, ndim):
        """
        Return an array containing the indices of the neighbors particles.

        :param indv: indice of the particle to update
        :param ndim: number of particles in the first dimension if toroidal grid is used
        """

        # Get the number of particles
        nindv = self.current.shape[0]

        # Full topology (including the particle itself)
        if topology == 'full':
            neighborhood = np.zeros(nindv, dtype=np.int16)
            for i in range(0, nindv):
                neighborhood[i] = i

        # Ring topology (including the particle itself)
        if topology == 'ring':
            neighborhood = np.zeros(3, dtype=np.int16)
            ineighbor = 0
            for i in range(indv-1, indv+2):
                neighborhood[ineighbor] = i
                if i < 0:
                    neighborhood[ineighbor] = nindv-1
                if i == nindv:
                    neighborhood[ineighbor] = 0
                ineighbor += 1

        # Toroidal topology (including the particle itself)
        if topology == 'toroidal':
            # If the number of particles is a multiple of ndim
            if nindv%ndim == 0:
                # Get grid size
                n1 = ndim
                n2 = int(nindv/ndim)
                neighborhood = np.zeros(5, dtype=np.int16)
                # Get the indice of the particle on the grid
                i2 = int(indv/ndim)
                i1 = int(indv-i2*n1)
                # Fill neighborhood
                neighborhood[0] = indv
                # Get the indice of the neighbors
                # top
                if i1 == 0:
                    neighborhood[1] = i2*n1+(n1-1)
                else:
                    neighborhood[1] = i2*n1+(i1-1)
                # right
                if i2 == n2-1:
                    neighborhood[2] = i1
                else:
                    neighborhood[2] = (i2+1)*n1+i1
                # bottom
                if i1 == n1-1:
                    neighborhood[3] = i2*n1
                else:
                    neighborhood[3] = i2*n1+(i1+1)
                # left
                if i2 == 0:
                    neighborhood[4] = ((n2-1)*n1)+i1
                else:
                    neighborhood[4] = (i2-1)*n1+i1

        return neighborhood

    def _get_grid(self, ndim):
        """
        Define toroidal grid

        :param ndim: number of particles in the first dimension of the toroidal grid.
        """

        # Get the number of particles
        nindv = self.current.shape[0]

        # If the number of particles is a multiple of ndim
        if nindv%ndim == 0:
            # Initialize grid dimensions
            n1 = ndim
            n2 = int(nindv/ndim)
            # Initialize neighbor array (4 neighbor per particle)
            vngrid = np.zeros((nindv, 4), dtype=np.int16)
            # Loop over toroidal grid dimensions
            for i2 in range(0, n2):
                for i1 in range(0, n1):
                    # Get the indice of the neighbors
                    k = (i2*n1)+i1
                    # top
                    if i1 == 0:
                        vngrid[k, 0] = i2*n1+(n1-1)
                    else:
                        vngrid[k, 0] = i2*n1+(i1-1)
                    # right
                    if i2 == n2-1:
                        vngrid[k, 1] = i1
                    else:
                        vngrid[k, 1] = (i2+1)*n1+i1
                    # bottom
                    if i1 == n1-1:
                        vngrid[k, 2] = i2*n1
                    else:
                        vngrid[k, 2] = i2*n1+(i1+1)
                    # left
                    if i2 == 0:
                        vngrid[k, 3] = ((n2-1)*n1)+i1
                    else:
                        vngrid[k, 3] = (i2-1)*n1+i1

        # The number of particles is not a multiple of ndim
        else:
            raise ValueError('ndim must be a multiple of nindv')

        return vngrid

    def get_gbest(self, topology, indv=0, ndim=0):
        """
        Get gbest particle of the whole swarm or in the neighborhood of
        a given particle.

        .. rubric:: Basic usage

        >>> topology = 'full'
        >>> best = population.get_gbest(topology)
        """

        nindv = self.current.shape[0]

        # Get the best particle of the whole swarm
        if topology == 'full':
            ibest = np.argmin(self.misfit[:])

        # Get the best particle in the neighborhood (1 left, 1 right)
        # of the particle including itself.
        if topology == 'ring':
            ibest = indv
            vbest = self.misfit[indv]
            for i in range(indv-1, indv+2):
                ii = i
                if i < 0:
                    ii = nindv-1
                if i == nindv:
                    ii = 0
                if self.misfit[ii] < vbest:
                    ibest = ii
                    vbest = self.misfit[ii]

        # Get the best particle in the neighborhood (1 left, 1 right)
        # of the particle excluding itself.
        if topology == 'ringx':
            ileft = indv-1
            iright = indv+1
            if indv == 0:
                ileft = nindv-1
            if indv == nindv-1:
                iright = 0
            if self.misfit[ileft] <= self.misfit[iright]:
                ibest = ileft
            else:
                ibest = iright

        # Get the best particle in the neighborhood (left, right, top, bottom)
        # of the particle including itself.
        if topology == 'toroidal':
            grid = self._get_grid(ndim)
            ibest = indv
            vbest = self.misfit[indv]
            for i in range(0, 4):
                if self.misfit[grid[indv, i]] < vbest:
                    ibest = grid[indv, i]
                    vbest = self.misfit[grid[indv, i]]

        # Get the best particle in the neighborhood (left, right, top, bottom)
        # of the particle excluding itself.
        if topology == 'toroidalx':
            # Get the grid
            grid = self._get_grid(ndim)
            # Initialize best particule to a virtual particle with a maximum misfit
            ibest = -1
            vbest = np.amax(self.misfit[:]*2.)
            for i in range(0, 4):
                if self.misfit[grid[indv, i]] < vbest:
                    ibest = grid[indv, i]
                    vbest = self.misfit[grid[indv, i]]

        return self.history[ibest, :, :]

    def update(self, **kwargs):
        """
        Standard PSO update.

        :param control: 0 for weight (default), 1 for constriction
        :param c_0: value of the control parameter (default 0.7298)
        :param c_1: value of the cognitive parameter (default 2.05)
        :param c_2: value of the social parameter (default 2.05)
        :param topology: used topology (default 'full'): full, ring, ringx, toroidal, toroidalx
        :param ndim: number of particles in the first dimension if toroidal topology is used
        :param pupd: parameter update probability
        """

        # Parse kwargs parameter list
        ctrl = kwargs.get('control', 0)
        omega = kwargs.get('c_0', 0.7298)
        topology = kwargs.get('topology', 'full')
        ndim = kwargs.get('ndim', 0)
        pupd = kwargs.get('pupd', 2.0)

        if ctrl == 0:
            cog = kwargs.get('c_1', 2.05)
            soc = kwargs.get('c_2', 2.05)
        if ctrl == 1:
            cog = omega*kwargs.get('c_1', 2.05)
            soc = omega*kwargs.get('c_2', 2.05)

        # Update process
        for indv in range(0, self.current.shape[0]):
            gbest = self.get_gbest(topology, indv, ndim)
            for ipts in range(0, self.pspace.shape[0]):
                # Test if parameter will be updated
                if np.random.random_sample() <= pupd:
                    for ipar in range(0, self.pspace.shape[1]):

                        # Get values
                        current = self.current[indv, ipts, ipar]
                        velocity = self.velocity[indv, ipts, ipar]
                        history = self.history[indv, ipts, ipar]

                        # Update velocity vector
                        self.velocity[indv, ipts, ipar] = omega*velocity\
                                                    + cog*np.random.random_sample()\
                                                    * (history-current)\
                                                    + soc*np.random.random_sample()\
                                                    * (gbest[ipts, ipar]-current)

                        # Check particle velocity
                        if(np.abs(self.velocity[indv, ipts, ipar]) > self.pspace[ipts, ipar, 2]):
                            self.velocity[indv, ipts, ipar] = \
                                np.sign(self.velocity[indv, ipts, ipar])\
                                * self.pspace[ipts, ipar, 2]

                        # Update particle position
                        self.current[indv, ipts, ipar] += self.velocity[indv, ipts, ipar]

                        # Check if particle is in parameter space
                        if(self.current[indv, ipts, ipar] < self.pspace[ipts, ipar, 0]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 0]
                        if(self.current[indv, ipts, ipar] > self.pspace[ipts, ipar, 1]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 1]

    def bbupdate(self, **kwargs):
        """
        Bare bone PSO update.

        :param topology: used topology (default 'full'): full, ring, ringx, toroidal, toroidalx
        :param ndim: number of particles in the first dimension if toroidal topology is used
        :param pupd: parameter update probability
        """

        # Parse kwargs parameter list
        topology = kwargs.get('topology', 'full')
        ndim = kwargs.get('ndim', 0)
        pupd = kwargs.get('pupd', 1.0)

        # Update process
        for indv in range(0, self.current.shape[0]):
            gbest = self.get_gbest(topology, indv, ndim)
            for ipts in range(0, self.pspace.shape[0]):
                # Test if parameter will be updated
                if np.random.random_sample() <= pupd:
                    for ipar in range(0, self.pspace.shape[1]):

                        # Get values
                        current = self.current[indv, ipts, ipar]
                        velocity = self.velocity[indv, ipts, ipar]
                        history = self.history[indv, ipts, ipar]

                        # Normal distribution parameters
                        loc = (gbest[ipts, ipar]+history)/2.
                        sca = np.abs(gbest[ipts, ipar]-history)
                        print(indv, loc, sca)
                        # Update position vector
                        if sca > 0. :
                            self.current[indv, ipts, ipar] = np.random.normal(loc=loc, scale=sca)

                        # Check if particle is in parameter space
                        if(self.current[indv, ipts, ipar] < self.pspace[ipts, ipar, 0]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 0]
                        if(self.current[indv, ipts, ipar] > self.pspace[ipts, ipar, 1]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 1]

    def bbupdate2(self, **kwargs):
        """
        Bare bone PSO update.

        :param topology: used topology (default 'full'): full, ring, ringx, toroidal, toroidalx
        :param ndim: number of particles in the first dimension if toroidal topology is used
        :param pupd: parameter update probability
        """

        # Parse kwargs parameter list
        topology = kwargs.get('topology', 'full')
        ndim = kwargs.get('ndim', 0)
        pupd = kwargs.get('pupd', 1.0)
        stag = kwargs.get('stag', 5.0)

        # Update process
        for indv in range(0, self.current.shape[0]):
            gbest = self.get_gbest(topology, indv, ndim)
            for ipts in range(0, self.pspace.shape[0]):
                # Test if parameter will be updated
                if np.random.random_sample() <= pupd:
                    for ipar in range(0, self.pspace.shape[1]):

                        # Get values
                        current = self.current[indv, ipts, ipar]
                        velocity = self.velocity[indv, ipts, ipar]
                        history = self.history[indv, ipts, ipar]

                        # Normal distribution parameters
                        loc = (gbest[ipts, ipar]+history)/2.
                        sca = np.abs(gbest[ipts, ipar]-history)

                        # Update position vector
                        if sca > 0. :
                            self.current[indv, ipts, ipar] = np.random.normal(loc=loc, scale=sca)

                        # Check if particle is in parameter space
                        if(self.current[indv, ipts, ipar] < self.pspace[ipts, ipar, 0]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 0]
                        if(self.current[indv, ipts, ipar] > self.pspace[ipts, ipar, 1]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 1]


    def fiupdate(self, **kwargs):
        """
        Fully Informed PSO update.

        :param control: 0 for weight (default), 1 for constriction
        :param c_0: value of the control parameter (default 0.7298)
        :param c_1: value of the acceleration parameter (default 4.1)
        :param topology: used topology (default 'full'): full, ring, toroidal
        :param ndim: number of particles in the first dimension if toroidal topology is used
        :param weight: weight to apply to neighbors: flat, misfit
        :param pupd: parameter update probability
        """

        # Parse kwargs parameter list
        ctrl = kwargs.get('control', 0)
        omega = kwargs.get('c_0', 0.7298)
        topology = kwargs.get('topology', 'full')
        ndim = kwargs.get('ndim', 0)
        weight = kwargs.get('weight', 'flat')
        pupd = kwargs.get('pupd', 1.0)

        if ctrl == 0:
            acc = kwargs.get('c_1', 4.10)
        if ctrl == 1:
            acc = omega*kwargs.get('c_1', 4.10)

        # Update process
        for indv in range(0, self.current.shape[0]):
            # Get the neighbourhood of the particle
            neighborhood = self._get_neighbors(topology, indv, ndim)
            for ipts in range(0, self.pspace.shape[0]):
                # Test if parameter will be updated
                if np.random.random_sample() <= pupd:
                    for ipar in range(0, self.pspace.shape[1]):

                        # Get values
                        current = self.current[indv, ipts, ipar]
                        velocity = self.velocity[indv, ipts, ipar]
                        history = self.history[indv, ipts, ipar]

                        # Update velocity vector
                        nneighbor = len(neighborhood)
                        w = np.zeros(nneighbor, dtype=np.float32)
                        if weight == 'flat':
                            w[:] = 1.
                        if weight == 'misfit':
                            for ineighbor in range(0, nneighbor):
                                ii = neighborhood[ineighbor]
                                w[ineighbor] = 1./self.misfit[ii]
                        pnum = 0.
                        pden = 0.
                        for ineighbor in range(0, nneighbor):
                            ii = neighborhood[ineighbor]
                            r = np.random.random_sample()/float(nneighbor)
                            pnum += r*acc*w[ineighbor]*(self.history[ii, ipts, ipar])
                            pden += r*acc*w[ineighbor] #r*acc*w[ineighbor]

                            self.velocity[indv, ipts, ipar] = omega*velocity+acc*((pnum/pden)-current)

                        # Check particle velocity
                        if(np.abs(self.velocity[indv, ipts, ipar]) > self.pspace[ipts, ipar, 2]):
                            self.velocity[indv, ipts, ipar] = \
                                np.sign(self.velocity[indv, ipts, ipar])\
                                * self.pspace[ipts, ipar, 2]

                        # Update particle position
                        self.current[indv, ipts, ipar] += self.velocity[indv, ipts, ipar]

                        # Check if particle is in parameter space
                        if(self.current[indv, ipts, ipar] < self.pspace[ipts, ipar, 0]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 0]
                        if(self.current[indv, ipts, ipar] > self.pspace[ipts, ipar, 1]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 1]
