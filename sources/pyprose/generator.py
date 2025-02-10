import numpy as np
from scipy.ndimage import gaussian_filter

class ModelGenerator:
    def __init__(self, n1: int, n2: int, dh: float):
        self.x_grid, self.z_grid = self._to_grid(n1, n2, dh)

    def _to_grid(self, n1: int, n2: int, dh: float) -> tuple[np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray]
            Coordinate grids
        """
        return np.meshgrid(np.arange(n2)*dh, np.arange(n1)*dh)

    def voronoi(self, xp: np.ndarray, zp: np.ndarray, val: np.ndarray, sigma=None) -> np.ndarray:
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
        sigma : float, optional
            Standard deviation of the Gaussian kernel, by default None

        Returns
        -------
        np.ndarray
            Interpolated values
        """
        dists = np.sqrt((self.x_grid[..., np.newaxis] - xp)**2 + (self.z_grid[..., np.newaxis] - zp)**2)
        nearest_indices = np.argmin(dists, axis=-1)
        model = val[nearest_indices]
        if sigma is not None:
            return gaussian_filter(model, sigma=sigma)
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    

    # Rule of thumd: sigma must be ~equal to the dominant wavelenght to preserve the model
    # Here the max velocity is 1200 m/s max freq is 45 (15*3) so sigma=4
    # Another use is average velocity/dominant frequency

    # Gaussian filter preserve geometries and velocities not inverse distance
    
    np.random.seed(1)

    n1, n2, dh = 101, 601, 0.5
    modgen = ModelGenerator(n1, n2, dh)

    xp = np.random.random(15)*float((n2-1)*dh)
    zp = np.random.random(15)*float((n1-2)*dh)
    values = np.random.random(15)*1000+200

    start = time.time()
    modvoronoi = modgen.voronoi(xp, zp, values)
    print("voronoi:: ", time.time()-start, flush=True)

    start = time.time()
    modvoronoismooth = modgen.voronoi(xp, zp, values, sigma=4)
    print("voronoi smooth:: ", time.time()-start, flush=True)

    start = time.time()
    modinvdist = modgen.inverse_distance(xp, zp, values, 8)
    print("inverse_distance:: ", time.time()-start, flush=True)

    plt.subplot(311)
    plt.imshow(modvoronoi, extent=[0, n2*dh, n1*dh, 0])
    plt.scatter(xp, zp, edgecolors='red', facecolors='none')

    plt.subplot(312)
    plt.imshow(modvoronoismooth, extent=[0, n2*dh, n1*dh, 0])
    plt.scatter(xp, zp, edgecolors='red', facecolors='none')

    plt.subplot(313)
    plt.imshow(modinvdist, extent=[0, n2*dh, n1*dh, 0])
    plt.scatter(xp, zp, edgecolors='red', facecolors='none')
    
    plt.show()