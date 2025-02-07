import numpy as np
from pyprose.disp import dispersion

class MASW:

    def __init__(self, dobs: np.array, acquipar: dict):
        self.dobs = dobs
        self.gobs = None
        self.diag = None
        self._acquipar = self._check_acquipar(acquipar)

    def _check_acquipar(self, params: dict):
        # Required parameters
        required_params = ['xs', 'zs', 'xr', 'zr', 'nt', 'dt', 't0', 'f0']
        if not all(param in params for param in required_params):
            raise ValueError("All required parameters must be supplied.")
        return params
    
    def diagram(self, wmin, wmax, vmin, vmax, dv):
        # Calculate offset
        offset = np.abs(self._acquipar['xr']-self._acquipar['xs'], dtype=np.float32)

        # Fourier transform
        gobs = np.fft.rfft(self.dobs, axis=0)
        
        # Frequencies
        freq = np.fft.rfftfreq(self._acquipar['nt'], d=self._acquipar['dt'])
        iwmin = np.argmin(np.abs(freq-wmin))
        iwmax = np.argmin(np.abs(freq-wmax))
        w = 2.*np.pi*freq[iwmin:iwmax+1]

        # Velocities
        v = np.arange(vmin, vmax+dv, dv, dtype=np.float32)

        diag = dispersion.compute_diagram(np.array(gobs[iwmin:iwmax+1, :], dtype=np.complex64), np.array(w, dtype=np.float32), np.array(v, dtype=np.float32), np.array(offset, dtype=np.float32))
        for iw in range(0, len(w)):
            diag[:, iw] = diag[:, iw]/np.amax(diag[:, iw])
        self.diag = diag

        return diag
    
    def normalize(self):
        nv, nw = self.diag.shape
        diagnorm = np.zeros((nv, nw), dtype=np.float32)
        for iw in range(0, nw):
            diagnorm = self.diag[:, iw]/np.amax(self.diag[:, iw])
        return diagnorm