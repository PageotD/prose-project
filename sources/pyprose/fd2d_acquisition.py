import numpy as np
from dataclasses import dataclass

class Acquisition:

    def __init__(self, srcpos: np.array, recpos: np.array):
        self.srcpos = srcpos
        self.recpos = recpos

    def isingrid(self, acqgrid: np.array, n1: int, n2: int):
        for ipos in acqgrid:
            if ipos[0] < 0 or ipos[0] >= n1 or ipos[1] < 0 or ipos[1] >= n2:
                return False
        return True

    def to_grid(self, n1: int, n2: int, dh: float):
        """
        Convert (x,z) receiver positions in extended gird index
        """
        srcgridpos = np.array(self.srcpos / dh, dtype=np.int16)
        recgridpos = np.array(self.recpos / dh, dtype=np.int16)

        #return srcgridpos, recgridpos
        if self.isingrid(srcgridpos, n1, n2) and self.isingrid(recgridpos, n1, n2):
            return srcgridpos, recgridpos
        
        raise Exception("Acquisition is out of grid")

if __name__ == "__main__":
    import time

    srcpos = np.array([[0.0, 0.0]], dtype=np.float32)
    recpos = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

    n1, n2, npml = 100, 100, 10
    dh = 0.1

    t = time.time()
    acq = Acquisition(srcpos, recpos)
    srcgridpos, recgridpos = acq.to_grid(n1, n2, dh)
    print(srcgridpos, recgridpos)
    print(f"Acquisition to grid: {time.time()-t} s")