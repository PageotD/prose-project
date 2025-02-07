import numpy as np


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


def inverse_distance(n1, n2, dh, pw, xp, zp, val):
    # Create grid coordinates
    x_grid, z_grid = np.meshgrid(np.arange(n2)*dh, np.arange(n1)*dh)

    # Compute squared distances to all scatter points
    dists = np.sqrt((x_grid[..., np.newaxis] - xp)**2 + (z_grid[..., np.newaxis] - zp)**2)
    weights = np.where(dists > 0.0, 1.0 / np.power(dists, pw), 1.0)
    num = np.sum(weights * val, axis=-1)
    den = np.sum(weights, axis=-1)

    return num / den

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    np.random.seed(1)

    xp = np.random.random(25)*300
    zp = np.random.random(25)*50
    values = np.random.random(25)*1000

    start = time.time()
    model = voronoi(51, 301, 1., xp, zp, values)
    print("voronoi:: ", time.time()-start, flush=True)

    plt.subplot(211)
    plt.imshow(model)
    plt.scatter(xp, zp)

    start = time.time()
    model1 = inverse_distance(51, 301, 1., 8., xp, zp, values)
    print("inverse_distance:: ", time.time()-start, flush=True)

    plt.subplot(212)
    plt.imshow(model1)
    plt.scatter(xp, zp)
    plt.show()

    start = time.time()
    for i in range(200):
        model1 = inverse_distance(51, 301, 1., 8., xp, zp, values)
    print("inverse_distance:: ", time.time()-start, flush=True)
    print("inverse_distance mean:: ", (time.time()-start)/100., flush=True)

    # plt.subplot(311)
    # plt.imshow(model1, vmin=0, vmax=1000)
    # plt.scatter(xp, zp)
    # plt.subplot(312)
    # plt.imshow(model2, vmin=0, vmax=1000)
    # plt.scatter(xp, zp)
    # plt.subplot(313)
    # plt.imshow(model3, vmin=0, vmax=1000)
    # plt.scatter(xp, zp)
    # plt.show()