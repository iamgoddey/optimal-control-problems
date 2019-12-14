import numpy as np
import scikits.bvp1lg.colnew as colnew
from scipy.integrate import simps, trapz

"""
Book Example 2.5
Opt Example 4 of optimal control tutorial.
    minimize J = 1/2 int_0^T x^2 dt, with T = 2
    subject to: x' = u, p' = -x, |u| <= 1, and
    q' = 0 together with conditions
    x(0) = 1, p(q) = 0, p(T) = 0
"""

T = 2.0
S = 1.5

def X(t):
    return np.where(t <= 1, 1-t, 0)

def P(t):
    return np.where(t <= 1, .5*(t-1)**2, 0)

def fsub(t, z):
    """ ODE's for states and costates """
    x, p, q = z
    u = np.where(t <= S, -1, 0)
    zero = np.zeros_like(t)
    w = np.where(t <= S, q/S, (T-q)/(T-S))
    return [w*u, -w*x, zero]

def dfsub(t, z):
    """ ODE's for states and costates """
    x, p, q = z
    u = np.where(t <= S, -1, 0)
    one = np.ones_like(t)
    zero = np.zeros_like(t)
    w = np.where(t <= S, q/S, (T-q)/(T-S))
    return [[ zero, zero,  u/S], 
            [   -w, zero, -x/S],
            [ zero, zero, zero]]

def gsub(z):
    """ The boundary conditions """
    # x(0) = 1, p(q) = 0 and  p(T) = 0.
    x, p, q = z
    return [x[0]-1, 
            p[1],
            p[2]]

def dgsub(z):
    """ The boundary conditions """
    x, p, q = z
    return [[1,0,0],
            [0,1,0], 
            [0,1,0]] 

def guess(t):
    x = np.where(t <= S, 1-t, 0)
    p = np.where(t <= S, 1-t, 0)
    q = np.ones(t.shape)
    z = np.array([x,p,q])
    return z, fsub(t,z)
    

# Initial guess for the solution
N = 5
degrees = [1, 1, 1]
boundary_points = [0, S, T]
#tin = [0, .4, .8, 1.5, T]
tin = np.linspace(0, T, N)

# solve the boundary value problem
tol = [1e-5, 1e-5, 1e-5]
solution = colnew.solve(
    boundary_points, degrees, fsub, gsub,
    dfsub=dfsub, dgsub=dgsub,
    is_linear=False, tolerances=tol, initial_guess=guess,
    #extra_fixed_points=[0.800],
    collocation_points=3, initial_mesh=tin,
    vectorized=True, maximum_mesh_size=50, verbosity=2)

print ('grid used within colnew')
tc = solution.mesh
n = solution.nmesh
print (n)
print (tc)
# refine the grid (doubling)
t = np.zeros(2*n-1)
for i in range(n-1):
    t[2*i] = tc[i]
    t[2*i+1] = .5*(tc[i]+tc[i+1])
t[-1] = tc[-1]
print (t)
x = solution(t)[:,0]
p = solution(t)[:,1]
q = solution(t)[-1,2]

print ('grid used within colnew')
print (solution.nmesh)
print (t)
for j in range(len(t)):
    if t[j] < S + 1e-6:
        t[j] = t[j]*q/S
    else:
        t[j] = q + (T-q)/(T-S)*(t[j]-S) 
print ('grid transformed')
print (t)

print ('t* = ', q)
print ('T = ', T)
print ('x(T) = ', x[-1])
J = simps(x*x,x=t)/2
print('J = ', J)


import matplotlib.pyplot as plt
plt.figure(3)
plt.plot(t,np.zeros(len(t)),'b-')
for j in range(len(t)):
    plt.plot([t[j],t[j]],[-.05,.05],'k-')
plt.title('Grid')
plt.savefig('Grid1.png')
plt.show()

plt.figure(1)
plt.plot(t, x,'-', label="$x(t)$")
plt.plot(t, p,'-', label="$p(t)$")
plt.text(.70,0.5,'x(t)= 1-t')
plt.text(.25,0.0,'p(t)= $(t-1)^2/2$')
f = 'Final cost is: J='+str(J)
plt.text(0.41,.8,f)
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('COLNEW for Optimal Control Problem, Example 3.2.0.3')
plt.savefig('OPT4.png')
plt.show()




#te = np.linspace(0, T, 41)
#plt.figure(2)
#plt.plot(te, X(te),'-', label="$x(t)$")
#plt.plot(te, P(te),'-', label="$p(t)$")
#plt.xlabel('time')
#plt.ylabel('states')
#plt.grid()
#plt.legend(framealpha=1, shadow=True)
#plt.title('Exact solution, Example 4')
#plt.show()



