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
    	w = (T-q2)/(T-S[1])
    else:
    	u = -.5*(p+3)
        w = (q2-q1)/(S[1]-S[0])
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
tin = [0., 0.3, S[0], 0.5, S[1], 1.2, 1.7, T] 

# solve the boundary value problem
tol = [1e-6, 1e-6, 1e-6, 1e-7, 0] # change
solution = colnew.solve(
    boundary_points, degrees, fsub, gsub,
    dfsub=None, dgsub=dgsub,
    is_linear=False, tolerances=tol, initial_guess=guess,
    collocation_points = 5, initial_mesh=tin, adaptive_mesh_selection=True,
    vectorized=False, maximum_mesh_size=100, verbosity=2)

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
q1 = solution(t)[-1,2]
q2 = solution(t)[-1,3]
J = solution(t)[-1,4]

print ('grid used within colnew')
print (solution.nmesh)
print (t)
for j in range(len(t)):
    if t[j] <= S[0] + 1e-6:
        t[j] = t[j]*q1/S[0]
    elif t[j] >= S[1] - 1e-6:
    	t[j] = q2 + (T-q2)/(T-S[1])*(t[j]-S[1]) 
    else:
        t[j] = q1 + (q2-q1)/(S[1]-S[0])*(t[j]-S[0]) 
print ('grid transformed')
print (t)

print ('t* = ', q1, q2)
print ('T = ', T)
print ('x(T) = ', x[-1])

# Calculating the optimal value
u = np.zeros_like(t)
for i in range(len(t)):
	if t[i]<=S[0]:
		u[i] = 2
	elif t[i]>=S[1]:
		u[i] = 0
	else:
		u[i] = -.5*(p[i]+3)		        
print('u(t)= \n' +str(u))
print('J = ', J)


import matplotlib.pyplot as plt
plt.figure(3)
plt.plot(t,np.zeros(len(t)),'b-')
for j in range(len(t)):
    plt.plot([t[j],t[j]],[-.05,.05],'k-')
plt.title('Grid')
plt.savefig('Grid3.png')
plt.show()

plt.figure(4)
plt.plot(t, u,'g-', label="$u(t)$")
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.title('Control Function ($u(t)$), Example 3.2.0.7')
plt.savefig('control2.png')
plt.show()


plt.figure(1)
plt.plot(t, x,'-', label="$x(t)$")
plt.plot(t, p,'-', label="$p(t)$")
plt.plot(t, u,'-', label="$u(t)$")
plt.text(1.25,11.0,'x(t)= $7\exp(t)-2$')
plt.text(1.0,-10.0,'p(t)= $2(1-\exp(2-t))$')
f = 'Final cost is: J='+str(J)
plt.text(0.2,25.0,f)
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('COLNEW, Example 3.2.0.7')
plt.savefig('OPT6.png')
plt.show()

