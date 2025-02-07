import numpy as np
import matplotlib.pyplot as plt

# Parameters
n1 = 41
n2 = 401
dh = 0.25

vsmin = 300.
vsmax = 600.
vpmin = vsmin * np.sqrt(3.)
vpmax = vsmax * np.sqrt(3.)

f0 = 25.
dt = 0.0001

# Check stability
fmax = f0 * 3.
lambda_min = vsmin / fmax
print("grid:", dh, lambda_min/10.)
print("time:", dt, 0.606*dh/vpmax)

# Create model
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
#plt.plot(xfull, yfull)
#plt.show()

vsmodel = np.zeros((n1, n2), dtype=np.float32)

for j in range(n2):
    for i in range(n1):
        y = float(i)*dh
        if y <= yfull[j]:
            vsmodel[i, j] = vsmin
        else:
            vsmodel[i, j] = vsmax
plt.imshow(vsmodel, aspect='auto')
plt.show()

np.save('s_shape_vs.npy', vsmodel)