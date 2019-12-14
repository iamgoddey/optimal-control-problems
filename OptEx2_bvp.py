import numpy as np
from scipy.integrate import solve_bvp, simps, trapz

"""
    OptEx1: Example 1 of optimal control Problem.
    min.:   J = int_0^2 x dt
    sub to: x' = u;  x(0) = 1
    The minimum principle gives the bvp problem:
    x' = u; p' = -1; with x(0) = 1, p(2) = 0
    where u = -1
    Note: lambda is renamed as p!
    """

def ode(t, y):
    u = -np.ones(len(t))
    return [u, u]

def bc(ya, yb):
    return [ya[0]-1, yb[1]]

# Initial guess for the solution
N = 5
xin = np.linspace(0, 2, N)
yin = np.zeros((2,N))   # trivial guess

# solve the boundary value problem
sol = solve_bvp(ode, bc, xin, yin)
x = sol.x
y = sol.y[0]
p = sol.y[1]

# Calculate u(t) from x,p
u = -np.ones((N))
# Calculate the cost
J = simps(y,x=x)
#J = trapz(y,x=x)

print('J', J)
print(yin)
print(u)

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(x, y,'-', label='$x(t)$', markersize=0.5)
plt.plot(x, p,'--', label='$p(t)$')
plt.plot(x, u,'r:', label='$u(t)$')
plt.text(.33,.2,'x(t)= 1 - t')
plt.text(.33,-.9, 'u(t)= -1')
plt.text(.23,1.3,'p(t) = 2 - t')
s = 'Final cost is: J = '+str(J)
plt.text(.41,.8,s)
plt.xlabel('time')
plt.ylabel('states')
plt.grid()
plt.legend(framealpha=1, shadow=True)
plt.title('Solve_bvp: Optimal Control Problem of memo0 Ex2')
plt.savefig('OptEx2_bvp.png')
plt.show()
