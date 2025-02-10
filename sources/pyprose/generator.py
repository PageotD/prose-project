import numpy as np

class ModelGenerator:
    def __init__(self, n1: int, n2: int, dh: float):
        self.n1 = n1
        self.n2 = n2
        self.dh = dh
        self.x_grid, self.z_grid = self._to_grid()

    def _to_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Function to create coordinate grids

        Parameters
        ----------
        n1 : int
            Number of points in the first dimension
        n2 : int
            Number of points in the second dimension
        dh : float
            Space sampling

        Returns
        -------
        x_grid, z_grid
            Coordinate grids
        """
        x_grid, z_grid = np.meshgrid(np.arange(self.n2)*self.dh, np.arange(self.n1)*self.dh)
        return x_grid, z_grid
    
    def voronoi(self, xp: np.ndarray, zp: np.ndarray, val: np.ndarray) -> np.ndarray:
        """
        Simple Voronoi 2D interpolation.

        Parameters
        ----------
        xp : np.ndarray
            x-coordinates of points to interpolate
        zp : np.ndarray
            z-coordinates of points to interpolate
        val : np.ndarray
            value of the points to interpolate

        Returns
        -------
        np.ndarray
            Interpolated values
        """
        dists = np.sqrt((self.x_grid[..., np.newaxis] - xp)**2 + (self.z_grid[..., np.newaxis] - zp)**2)
        nearest_indices = np.argmin(dists, axis=-1)
        model = val[nearest_indices]

        return model    

    def inverse_distance(self, xp: np.ndarray, zp: np.ndarray, val: np.ndarray, pw: int) -> np.ndarray:
        """
        Inverse distance weightning 2D interpolation.

        Parameters
        ----------
        xp : np.ndarray
            x-coordinates of points to interpolate
        zp : np.ndarray
            z-coordinates of points to interpolate
        val : np.ndarray
            value of the points to interpolate
        pw : int
            power to apply

        Returns
        -------
        np.ndarray
            Interpolated values
        """
        dists = np.sqrt((self.x_grid[..., np.newaxis] - xp)**2 + (self.z_grid[..., np.newaxis] - zp)**2)
        weights = np.where(dists > 0.0, 1.0 / np.power(dists, pw), 1.0)
        num = np.sum(weights * val, axis=-1)
        den = np.sum(weights, axis=-1)

        return num / den
    
    def sibsons(self, xp: np.ndarray, zp: np.ndarray, val: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
def voronoi(n1, n2, dh, xp, zp, val):
    """
    Optimized Voronoi 2D interpolation using NumPy vectorization.
    """
    # Create coordinate grids
    x_grid, z_grid = np.meshgrid(np.arange(n2)*dh, np.arange(n1)*dh)

    # Compute squared distances to all scatter points
    dists = np.sqrt((x_grid[..., np.newaxis] - xp)**2 + (z_grid[..., np.newaxis] - zp)**2)

    # Find the index of the nearest point for each grid cell
    nearest_indices = np.argmin(dists, axis=-1)

    # Assign values from the nearest scatter points
    model = val[nearest_indices]

    return model

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    np.random.seed(1)

    n1, n2, dh = 101, 601, 0.5
    modgen = ModelGenerator(n1, n2, dh)

    xp = np.random.random(15)*float((n2-1)*dh)
    zp = np.random.random(15)*float((n1-2)*dh)
    values = np.random.random(15)*1000

    start = time.time()
    modvoronoi = modgen.voronoi(xp, zp, values)
    print("voronoi:: ", time.time()-start, flush=True)

    start = time.time()
    modinvdist = modgen.inverse_distance(xp, zp, values, 8)
    print("inverse_distance:: ", time.time()-start, flush=True)

    plt.subplot(211)
    plt.imshow(modvoronoi, extent=[0, n2*dh, n1*dh, 0])
    plt.scatter(xp, zp, edgecolors='red', facecolors='none')

    plt.subplot(212)
    plt.imshow(modinvdist, extent=[0, n2*dh, n1*dh, 0])
    plt.scatter(xp, zp, edgecolors='red', facecolors='none')
    
    plt.show()