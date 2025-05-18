import numpy as np

def F(x):
  return np.array([np.cos(x[0])-np.sin(x[1]), np.cos(x[0])])

def DF(x):
  dx = np.array([10**(-6), 0])
  dFdx = (F(x+dx) - F(x))*(10**6)

  dy = np.array([0, 10**(-6)])
  dFdy = (F(x+dy) - F(x))*(10**6)

  return np.array([
    dFdx,
    dFdy
  ])

x = np.array([1, 1])
while np.linalg.norm(F(x)) > 10**(-6):
  x1 = np.linalg.solve(DF(x), -F(x))
  x = x1 + x

print(f"x={x}, F(x)={F(x)}")