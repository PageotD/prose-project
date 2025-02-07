import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import Rbf

def curve():
    x1 = 30.0
    x2 = 70.0
    y1 = 3.
    y2 = 7.
    xsegment1 = np.arange(0, x1, dh)
    ysegment1 = np.ones(len(xsegment1)) * 3.
    xsegment2 = np.arange(x2+dh, float(n2-1)*dh+dh, dh)
    ysegment2 = np.ones(len(xsegment1)) * 7.

    xcurve = np.arange(x1, x2+dh, dh)
    ycurve = np.cos((x2-xcurve)/(x2-x1)*np.pi)
    ycurve = y1+(ycurve+1.)/2.*(y2-y1)


    xfull = np.concatenate((xsegment1, xcurve, xsegment2))
    yfull = np.concatenate((ysegment1, ycurve, ysegment2))
    return xfull, yfull

def genmodel(xp, zp, values, n1, n2):
    # Generate a smooth model using set of points

    # Add a random epsilon value to avoid singular matrix
    xp += np.random.normal(0, 0.01, size=len(xp))
    zp += np.random.normal(0, 0.01, size=len(zp))
    
    # Create a grid of coordinates for the fine grid
    x_fine = np.linspace(np.min(xp), np.max(xp), n2)
    z_fine = np.linspace(np.min(zp), np.max(zp), n1)
    x_fine_grid, z_fine_grid = np.meshgrid(x_fine, z_fine)

    # Perform RBF interpolation
    epsilon = 1e-10
    rbf = Rbf(xp, zp, values, function='linear', epsilon=epsilon, smooth=0)
    interpolated_values = rbf(x_fine_grid, z_fine_grid)

    return np.array(interpolated_values, dtype=np.float32)


iter = sys.argv[1]
with open(f"data/iter_{str(iter).zfill(3)}.json", "r") as f:
    data = json.load(f)

vs = np.load("s_shape/s_shape_vs.npy")

n1 = 41
n2 = 401
dh = 0.25
vmin = 200.
vmax = 700.

xcurve, ycurve = curve()

misfit = 100000.
mbest = None
mmean = None

misfit_tot = 0
for particle in data:
    gmodel = genmodel(particle['xp'], particle['zp'], particle['vs'], n1, n2)
    if particle['misfit'] < misfit:
        misfit = particle['misfit']
        mbest = gmodel
        xpbest = particle['xp']
        zpbest = particle['zp']
        vsbest = particle['vs']
    if mmean is None:
        mmean = gmodel #* 1./misfit
    else:
        mmean += gmodel #* 1./misfit
    misfit_tot += 1./misfit
mmean /= float(len(data))
print(np.amax(mmean), np.amin(mmean), misfit_tot)



# plt.imshow(mbest, aspect='auto', vmin=vmin, vmax=vmax, extent=[0., float(n2-1)*dh, float(n1-1)*dh, 0.])
# plt.colorbar()
# plt.plot(xcurve, ycurve, c="black", linestyle=":")
# plt.scatter(xpbest, zpbest, c=vsbest, s=25, ec='k')
# plt.title("Best model")
# plt.show()

# plt.imshow(mmean, aspect='auto', vmin=vmin, vmax=vmax, extent=[0., float(n2-1)*dh, float(n1-1)*dh, 0.])
# plt.colorbar()
# plt.plot(xcurve, ycurve, c="black", linestyle=":")
# plt.title("Mean model")
# plt.show()


fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(6, 8)

ax1 = fig.add_subplot(gs[:2,:4])
ax1.imshow(mbest, aspect='auto', vmin=vmin, vmax=vmax, extent=[0., float(n2-1)*dh, float(n1-1)*dh, 0.])
ax1.plot(xcurve, ycurve, c="black", linestyle=":")
ax1.scatter(xpbest, zpbest, c=vsbest, s=25, ec='k')
ax1.set_title("Best model")

ax2 = fig.add_subplot(gs[:2, 4:])
ax2.imshow(mmean, aspect='auto', vmin=vmin, vmax=vmax, extent=[0., float(n2-1)*dh, float(n1-1)*dh, 0.])
ax2.plot(xcurve, ycurve, c="black", linestyle=":")
ax2.set_title("Mean model")

count=0
for i in range(40,321,40):
    z = np.linspace(0, float(n1-1)*dh+1, n1)
    axn = fig.add_subplot(gs[3:, count])
    axn.set_ylim(z[-1], z[0])
    axn.plot(vs[:, i],z,c='black')
    axn.plot(mbest[:, i],z,c='green')
    axn.plot(mmean[:, i],z,c='red')
    axn.set_title(f"{float(i)*dh} m")
    count+=1

plt.show()
