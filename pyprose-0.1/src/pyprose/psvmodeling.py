import numpy as np
from pyprose.fd2d import elasticproperties
from pyprose.fd2d import acquisition
from pyprose.fd2d import wavefields

class PSVModeling:

    def __init__(self, elastparin: dict=None, acquiparin: dict=None):
        self._elastpar = None
        self._acquipar = None

        if elastparin is not None:
            self._elastpar = self._check_elastpar(elastparin)

        if acquiparin is not None:
            self._acquipar = self._check_acquipar(acquiparin)

    def _check_elastpar(self, params: dict):
        # Required parameters
        required_params = ['vp', 'vs', 'rho', 'dh']
        if not all(param in params for param in required_params):
            raise ValueError("All required parameters must be supplied.")
        # Check optional parameters
        params.update({
            'surf': params.get('surf', 0),
            'npml': params.get('npml', 10),
            'apml': params.get('apml', 800.),
            'ppml': params.get('ppml', 3)
        })
        return params

    def _check_acquipar(self, params: dict):
        # Required parameters
        required_params = ['xs', 'zs', 'xr', 'zr', 'nt', 'dt', 't0', 'f0']
        if not all(param in params for param in required_params):
            raise ValueError("All required parameters must be supplied.")
        return params
        
    def run(self):
        n1 = self._elastpar['n1']
        n2 = self._elastpar['n2']
        dh = self._elastpar['dh']
        npml = self._elastpar['npml']
        apml = self._elastpar['apml']
        ppml = self._elastpar['ppml']
        isurf = self._elastpar['isurf']

        # Extend models
        vpe = elasticproperties.addpmls(n1, n2, npml, self._elastpar['vp'])
        vse = elasticproperties.addpmls(n1, n2, npml, self._elastpar['vs'])
        rhoe = elasticproperties.addpmls(n1, n2, npml, self._elastpar['rho'])

        # Buoyancy & Lame
        bux, buz = elasticproperties.buoyancy(n1+2*npml, n2+2*npml, rhoe)
        mu, lbd, lbdmu = elasticproperties.lame(n1+2*npml, n2+2*npml, vpe, vse, rhoe)

        # PMLs
        pmlx0, pmlx1, pmlz0, pmlz1 = elasticproperties.pmlgrids(n1, n2, dh, isurf, npml, ppml, apml)

        # Acquisition
        acq = np.column_stack((self._acquipar['xr'], self._acquipar['zr']))
        acqui = acquisition.cacqpos(n1, n2, dh, npml, acq)

        ricker = acquisition.cricker(self._acquipar['nt'], self._acquipar['dt'], self._acquipar['f0'], self._acquipar['t0'])
        gridsrc = acquisition.csrcspread(n1, n2, dh, npml, self._acquipar['xs'], self._acquipar['zs'], 1)

        ux, uz, wavex, wavez = wavefields.evolution(mu, lbd, lbdmu, bux, buz, pmlx0, pmlx1, pmlz0, pmlz1, npml, isurf, 2, ricker, gridsrc, acqui, dh, self._acquipar['nt'], self._acquipar['dt'])
    
        return wavex, wavez
    
    @property
    def elastpar(self):
        return self._elastpar
    
    @elastpar.setter
    def elastpar(self, param: dict):
        self._elastpar = self._check_elastpar(param)

    @property
    def acquipar(self):
        return self._acquipar
    
    @elastpar.setter
    def acquipar(self, param: dict):
        self._acquipar = self._check_acquipar(param)