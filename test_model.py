import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def genmodel(xp, zp, values, n1, n2):
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


xp = np.random.random(10)*500
zp = np.random.random(10)*100
values = np.random.random(10)*1000

model = genmodel(xp, zp, values, 100, 500)

plt.imshow(model)
plt.scatter(xp, zp)
plt.show()