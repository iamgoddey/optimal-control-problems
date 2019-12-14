import numpy as np
import scikits.bvp1lg.colnew as colnew
from scipy.integrate import simps, trapz

"""
Try Example 1
Opt Example 5 of optimal control tutorial.
    minimize J = int_0^T (u^2 + 3u - 2x) dt, with T = 1
    subject to: x' = x+u, p' = 2-p, u \in [0,2], and
    q' = 1 together with conditions
    x(0) = 5, p(q) + 3 = 0 and p(T) = 0
"""

T = 1.0
S = 1 - np.log(2.5)

def X(t):
    return 5*np.exp(t)

def P(t):
    return  2*(1-np.exp(1-t))

def fsub(t, z):
    """ ODE's for states and costates """
    x, p, q, j = z
    u = np.where(t<= S, -.5*(p+3), 0)
    zero = np.zeros_like(t)
    w = np.where(t <= S, q/S, (T-q)/(T-S))
    return [w*(x+u), w*(2-p), zero, w*(u**2 + 3*u - 2*x)]

def dfsub(t, z):
    """ ODE's for states and costates """
    x, p, q, j = z
    u = np.where(t <= S, -.5*(p+3), 0)
    zero = np.zeros_like(t)
    w = np.where(t <= S, q/S, (T-q)/(T-S))
    return [[ w, zero,  (x+u)/S, zero], 
            [ zero, -w, (2-p)/S, zero], 
            [ zero, zero, zero, zero],
            [-2*w, zero, zero, zero]]

def gsub(z):
    """ The boundary conditions """
    # j(0)=0, x(0) = 5 , p(q) = -3 an p(T) = 0.
    x, p, q,  j = z
    return [j[0], 
    		x[1]-5, 
    		p[2]+3,
            p[3]]

def dgsub(z):
    """ The boundary conditions """
    x, p, q , j= z
    return [[0,0,0,1],
    		[1,0,0,0],
            [0,1,0,0], 
            [0,1,0,0]] 

def guess(t):
    x = 5*np.exp(t)
    p = 2*(1-np.exp(1-t))
    q = S*np.ones_like(t)
    j = -2*x
    z = np.array([x,p,q,j])
    return z, fsub(t,z)
    

# Initial guess for the solution
N = 5
degrees = [1, 1, 1, 1]
boundary_points = [0, 0, S, T]
tin = [0., S, .2, .5, 0.8, T]

# solve the boundary value problem
tol = [1e-6, 1e-6, 1e-5, 0]
solution = colnew.solve(
    boundary_points, degrees, fsub, gsub,
    dfsub=dfsub, dgsub=dgsub,
    is_linear=False, tolerances=tol, initial_guess=guess,
    collocation_points=5, initial_mesh=tin, adaptive_mesh_selection=True,
    vectorized=True, maximum_mesh_size=100, verbosity=2)

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
u = np.where(t<= S, -.5*(p+3), 0)
v = u**2 + 3*u - 2*x 
J = simps(v,x=t)
print('J = ', J)


import matplotlib.pyplot as plt
plt.figure(3)
plt.plot(t,np.zeros(len(t)),'b-')
for j in range(len(t)):
    plt.plot([t[j],t[j]],[-.05,.05],'k-')
plt.title('Grid')
plt.savefig('Grid2.png')
plt.show()

plt.figure(1)
plt.plot(t, x,'-', label="$x(t)$")
plt.plot(t, p,'-', label="$p(t)$")
plt.plot(t, u,'-', label="$u(t)$")
plt.text(.70,8.5,'x(t)= $5\exp(t)$')
plt.text(0.5,-2.5,'p(t)= $2(1-\exp(1-t))$')
f = 'Final cost is: J='+str(J)
plt.text(0.25,2.8,f)
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('COLNEW, Example 3.2.0.5')
plt.savefig('OPT5.png')
plt.show()


plt.figure(4)
plt.plot(t, u,'g-', label="$u(t)$")
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.title('Control Function ($u(t)$), Example 3.2.0.5')
plt.savefig('Control5.png')
plt.show()

#te = np.linspace(0, T, 41)
#plt.figure(2)
#plt.plot(te, X(te),'-', label="$x(t)$")
#plt.plot(te, P(te),'-', label="$p(t)$")
#plt.plot(t, u,'-', label="$u(t)$")
#plt.text(.70,8.5,'x(t)= $5\exp(t)$')
#plt.text(0.5,-2.5,'p(t)= $2(1-\exp(1-t))$')
#f = 'Final cost is: J='+str(J)
#plt.text(0.25,2.8,f)
#plt.xlabel('time')
#plt.ylabel('states')
#plt.grid()
#plt.legend(framealpha=1, shadow=True)
#plt.title('Exact solution, Example 3.2.5')
#plt.show()
