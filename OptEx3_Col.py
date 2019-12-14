import numpy as np
import scikits.bvp1lg.colnew as colnew
from scipy.integrate import simps, trapz

"""
    OptEx1: Example 1 of optimal control Problem.
    min.:   J = 0.5 * int_0^1 x**2 dt
    sub to: x' = u;  x(0) = 1
    The minimum principle gives the bvp problem:
    x' = u; p' = -x ; with x(0) = 1, p(1) = 0
    where u = -1
    Note: lambda is renamed as p!
    """

def fsub(t, z):
    u = -np.ones(len(t))
    return np.array([u, -z[0]*np.ones(len(t))])

def gsub(z):
    x, p = z
    return np.array([x[0]-1, p[1]])

def guess(t):
    x = 2*np.ones_like(t)
    p = -.5*np.ones_like(t)
    z = np.array([x,p])
    dm = fsub(t,z)
    return z, dm

# Initial guess for the solution
N = 11
degrees = [1, 1]
boundary_points = [0, 1]
tin = np.linspace(0, 1, N)

# solve the boundary value problem
tol = [1e-9, 1e-9]
solution = colnew.solve(
    boundary_points, degrees, fsub, gsub,
    dfsub=None, dgsub=None,
    is_linear=False, tolerances=tol, initial_guess=guess,
    collocation_points=3, initial_mesh=tin,
    vectorized=True, maximum_mesh_size=30, verbosity=0)

t = solution.mesh
x = solution(t)[:,0]
p = solution(t)[:,1]

# Calculate u(t) from x,p
u = -np.ones(len(t))
# Calculate the cost
w = 0.5*(1-t)**2
J = simps(w,x=t)
#J = trapz(x,x=t)

print('J', J)
print('u', u)
print('p', p)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(t, x,'-', label='$x(t)$')
plt.plot(t, p,'--', label='$p(t)$', markersize=0.5)
plt.plot(t, u,'r:', label='$u(t)$')
plt.text(.23,0.5,'x(t)= t - 1')
plt.text(.33,-.9, 'u(t)= -1')
plt.text(.43,-0.1,'p(t)= 0.5 t^2 - t + 0.5')
s = 'Final cost is: J = '+str(J)
#s = 'Final cost is: J = '+str(np.round(J))
plt.text(0.41,.8,s)
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('colnew: Optimal control Problem of memo0 Ex3')
plt.savefig('optEx3_Col.png')
plt.show()
