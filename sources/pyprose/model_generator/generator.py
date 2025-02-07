import numpy as np

def voronoi(n1, n2, dh, xp, zp, val):
    """
    Simple Voronoi 2D interpolation.
    """

    model = np.zeros((n1, n2), dtype=np.float32)

    for i2 in range(0, n2):
        x = float(i2)*dh
        for i1 in range(0, n1):
            z = float(i1)*dh
            # Initialize minimum distance
            dmin = 0.
            # Loop over number of scatter points
            for ipts in range(0, np.size(xp)):
                d = np.sqrt((x - xp[ipts])**2 + (z - zp[ipts])**2)
                if ipts == 0:
                    dmin = d
                    imin = ipts
                elif d < dmin:
                    dmin = d
                    imin = ipts
            model[i1, i2] = val[imin]

    return model

def inverse_distance(n1, n2, dh, pw, xp, zp, val):
    """
    Inverse distance weightning 2D interpolation.
    """

    model = np.zeros((n1, n2), dtype=np.float32)

    for i2 in range(0, n2):
        x = float(i2)*dh
        for i1 in range(0, n1):
            z = float(i1)*dh
            # Initialize numerator and denominator
            num = 0.
            den = 0.
            # Loop over number of scatter points
            for ipts in range(0, np.size(xp)):
                d = np.sqrt((x - xp[ipts])**2 + (z - zp[ipts])**2)
                if np.power(d, pw) > 0.0:
                    w = 1.0 / np.power(d, pw)
                    num += w * val[ipts]
                    den += w
                else:
                    num = val[ipts]
                    den = 1.
            model[i1, i2] = num / den

    return model

def inverse_distance2(n1, n2, dh, pw, xp, zp, val):
    # Create grid coordinates
    x_grid, z_grid = np.meshgrid(np.arange(n2)*dh, np.arange(n1)*dh)
    # Reshape scatter points and values for broadcasting
    xp = xp.reshape(-1, 1, 1)
    zp = zp.reshape(1, -1, 1)
    val = val.reshape(1, 1, -1)
    # Calculate distance
    d = np.sqrt((x_grid - xp)**2 + (z_grid - zp)**2)
    # Calculate weights using inverse distance
    w = np.power(d, -pw, where=(d !=0))
    # Handle zero distances
    zd = (d == 0)
    w = np.where(zd, 1, w)
    # Calculate w sum and sum of w
    wsum = np.sum(w*val, axis=0)
    sumw = np.sum(w, axis=0)

    # Calculate interpolated values
    model = wsum / sumw

    return model.astype(np.float32)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    np.random.seed(1)

    xp = np.random.random(20)*300
    zp = np.random.random(20)*50
    values = np.random.random(20)*1000

    start = time.time()
    model = voronoi(51, 301, 1., xp, zp, values)
    print("voronoi:: ", time.time()-start, flush=True)

    plt.imshow(model)
    plt.scatter(xp, zp)
    plt.show()

    start = time.time()
    model1 = inverse_distance2(51, 301, 1., 2., xp, zp, values)
    print("inverse_distance 1:: ", time.time()-start, flush=True)

    start = time.time()
    model2 = inverse_distance2(51, 301, 1., 4., xp, zp, values)
    print("inverse_distance 2:: ", time.time()-start, flush=True)

    start = time.time()
    model3 = inverse_distance2(51, 301, 1., 8., xp, zp, values)
    print("inverse_distance 3:: ", time.time()-start, flush=True)

    plt.subplot(311)
    plt.imshow(model1, vmin=0, vmax=1000)
    plt.scatter(xp, zp)
    plt.subplot(312)
    plt.imshow(model2, vmin=0, vmax=1000)
    plt.scatter(xp, zp)
    plt.subplot(313)
    plt.imshow(model3, vmin=0, vmax=1000)
    plt.scatter(xp, zp)
    plt.show()