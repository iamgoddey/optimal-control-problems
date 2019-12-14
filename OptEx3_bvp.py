import numpy as np
from scipy.integrate import solve_bvp, simps, trapz

"""
    OptEx1: Example 1 of optimal control Problem.
    min.:   J = 0.5 int_0^1 x**2 dt
    sub to: x' = u;  x(0) = 1, u \in Omega = [0,1]
    The minimum principle gives the bvp problem:
    x' = u; p' = -x; with x(0) = 1, p(1) = 0
    where u = -1
    Note: lambda is renamed as p!
    """

def ode(t, y):
    u = -np.ones(len(t))
    return [u, -y[0]*np.ones(len(t))]

def bc(ya, yb):
    return [ya[0]-1, yb[1]]

# Initial guess for the solution
N = 11
xin = np.linspace(0, 1, N)
yin = np.zeros((2,N))   # trivial guess

# solve the boundary value problem
sol = solve_bvp(ode, bc, xin, yin)
x = sol.x
y = sol.y[0]
p = sol.y[1]

# Calculate u(t) from x,p
u = -np.ones((N))
# Calculate the cost
w = 0.5*(1-x)**2
J = simps(w,x=x)
#J = trapz(w,x=x)

print('J', J)
print(yin)
print(u)

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(x, y,'-', label='$x(t)$', markersize=0.5)
plt.plot(x, p,'--', label='$p(t)$')
plt.plot(x, u,'r:', label='$u(t)$')
plt.text(.33,.4,'x(t)= t - 1')
plt.text(.33,-.9, 'u(t)= -1')
plt.text(.23,-0.1,'p(t) = 0.5 t^2 - t + 0.5')
s = 'Final cost is: J = '+str(J)
plt.text(.41,.8,s)
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('Solve_bvp: Optimal Control Problem of memo0 Ex3')
plt.savefig('optEx3_bvp.png')
plt.show()
