import numpy as np

a = np.array([-1, -1, -1, 1, 1, -1, 1, 1])
b = np.array([-1, -1, 1, -1, 1, 1, 1, -1])
c = np.array([-1, 1, 1, 1, 1, 1, -1, -1])
d = np.array([-1, 1, -1, -1, -1, -1, 1, -1])
x = np.array([-1, 1, -3, 1, -1, -3, 1, 1])

print("a*x=", a.dot(x.T))
print("b*x=", b.dot(x.T))
print("c*x=", c.dot(x.T))
print("d*x=", d.dot(x.T))
