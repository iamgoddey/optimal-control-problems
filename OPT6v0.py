import numpy as np
import scikits.bvp1lg.colnew as colnew
from scipy.integrate import simps, trapz

"""
Try Example 1
Opt Example 5 of optimal control tutorial.
    minimize J = int_0^T (u^2 + 3u - 2x) dt, with T = 2
    subject to: x' = x + u, p' = 2-p, u \in [0,2], and
    q' = 1 together with conditions
    x(0) = 5, p(q) + 7 = 0, p(q) + 3 = 0 and p(T) = 0
"""

T = 2.0
S = [2 - np.log(4.5), 2 - np.log(2.5)]

def X(t):
    return 7*np.exp(t)-2

def P(t):
    return 2*(1-np.exp(2-t))

def fsub(t, z):
    """ ODE's for states and costates """
    x, p, q1, q2, j = z
    if t <= S[0]:
    	u = 2
        w = q1/S[0]
    elif t>= S[1]:
    	u = 0
    	w = q2 + (T-q2)/(T-S[1])*(t-S[1])
    else:
    	u = -.5*(p+3)
        w = q1 + (q2-q1)/(S[1]-S[0])*(t-S[1]) 
    return [w*(x+u), w*(2-p), 0, 0, w*(u**2 + 3*u - 2*x)]


def gsub(z):
    """ The boundary conditions """
    # j(0)=0, x(0) = 5 , p(q) = -7, p(q) = -3 an p(T) = 0.
    x, p, q1, q2, j = z
    return [j[0],   
    		x[1]-5, 
    		p[2]+7,
            p[3]+3,
            p[4]]

def dgsub(z):
    """ The boundary conditions """
    x, p, q1, q2, j= z
    return [[0,0,0,0,1],
    		[1,0,0,0,0],
            [0,1,0,0,0],
            [0,1,0,0,0], 
            [0,1,0,0,0]] 

def guess(t):
    x = 7*np.exp(t)- 2
    p = 2*(1-np.exp(2-t))
    q1 = 0.5
    q2 = 0.8
    j = -2*x
    z = np.array([x,p,q1,q2,j])
    return z, fsub(t,z)
    

# Initial guess for the solution
N = 10
degrees = [1, 1, 1, 1, 1]
boundary_points = [0, 0, S[0], S[1], T] 
tin = [0., 0.2, 0.4, S[0], 0.5, S[1], 1.4, T] 

# solve the boundary value problem
tol = [1e-6, 1e-6, 1e-5, 1e-5, 0] # change
solution = colnew.solve(
    boundary_points, degrees, fsub, gsub,
    dfsub=None, dgsub=dgsub,
    is_linear=False, tolerances=tol, initial_guess=guess,
    collocation_points = 5, initial_mesh=tin, adaptive_mesh_selection=True,
    vectorized=False, maximum_mesh_size=50, verbosity=2)

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
j = solution(t)[-1,3]

print ('grid used within colnew')
print (solution.nmesh)
print (t)
for j in range(len(t)):
    if t[j] < S[0] + 1e-6:
        t[j] = t[j]*q/S[0]
    else:
        t[j] = q + (T-q)/(T-S[1])*(t[j]-S[1]) 
print ('grid transformed')
print (t)

print ('t* = ', q)
print ('T = ', T)
print ('x(T) = ', x[-1])

# Calculating the optimal value

#u = np.where(t<= S, -.5*(p+3), 0)
#v = u**2 + 3*u - 2*x 
#J = simps(v,x=t)
#print('J = ', J)


import matplotlib.pyplot as plt
plt.figure(3)
plt.plot(t,np.zeros(len(t)),'b-')
for j in range(len(t)):
    plt.plot([t[j],t[j]],[-.05,.05],'k-')
plt.title('Grid')
plt.show()

plt.figure(1)
plt.plot(t, x,'-', label="$x(t)$")
plt.plot(t, p,'-', label="$p(t)$")
#plt.plot(t, u,'-', label="$u(t)$")
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('colnew, Example 6')
plt.show()

#te = np.linspace(0, T, 41)
#plt.figure(2)
#plt.plot(te, X(te),'-', label="$x(t)$")
#plt.plot(te, P(te),'-', label="$p(t)$")
#plt.xlabel('time')
#plt.ylabel('states')
#plt.grid()
#plt.legend(framealpha=1, shadow=True)
#plt.title('exact solution, Example 6')
#plt.show()
