# optimal-control-problems
Numerical Methods for Optimal Control Problems:

The solution to optimal control problems for ordinary differential equations can be obtained by applying Pontryagin's minimum principle. This usually yields in general a non-linear boundary value problem which has to be solved numerically. Such methods are called indirect methods as they are solving first order necessary conditions. The boundary value problems to be solved are of the form  <img src="https://latex.codecogs.com/gif.latex?\[&space;y'=&space;f(t,y,u),&space;\quad&space;0\le&space;t&space;\le&space;t_f&space;\]" title="\[ y'= f(t,y,u), \quad 0\le t \le t_f \]" />  with the set of boundary conditions expressed in the form  <img src="https://latex.codecogs.com/gif.latex?\[g_1(y(0))&space;=&space;0&space;\mbox{&space;and&space;}&space;g_2(y(t_f))&space;=&space;0\]" title="\[g_1(y(0)) = 0 \mbox{ and } g_2(y(t_f)) = 0\]" />   where <img src="https://latex.codecogs.com/gif.latex?$&space;y&space;$" title="$ y $" /> is the vector function of the state and co-state variables, <img src="https://latex.codecogs.com/gif.latex?$&space;u&space;$" title="$ u $" /> is the scalar or vector function of control variables, <img src="https://latex.codecogs.com/gif.latex?$&space;g_1&space;\in&space;\RR^n&space;\mbox{&space;and&space;}&space;g_2&space;\in&space;\RR^m&space;$" title="$ g_1 \in \RR^n \mbox{ and } g_2 \in \RR^m $" />  for some values of <img src="https://latex.codecogs.com/gif.latex?$&space;m&space;\mbox{&space;and&space;}&space;n&space;\mbox{&space;with&space;}&space;1&space;\le&space;m&space;<&space;n$" title="$ m \mbox{ and } n \mbox{ with } 1 < m < n$" />  where each vector functions <img src="https://latex.codecogs.com/gif.latex?$&space;g_1&space;\mbox{&space;and&space;}&space;g_2$" title="$ g_1 \mbox{ and } g_2$" /> are independent. The boundary value problem also requires the satisfaction of two-point or multi-point boundary conditions. Of special interest are optimal control problems with constraints either for the control or the state variables. For such problems, the right hand side of the differential equation may be piecewise smooth, that is, there are points at which the right $f(t,y,u)$ jumps as the control variable $u$ may show discontinuities.

The non-smooth behaviour of the right hand side inhibits a reliable convergence of the numerical approximations towards the exact solution. A way around this problem is a transformation of the points with non-smooth behaviour to known, fixed locations $t_i \in [0, t_f] $. Then it is obvious that the numerical approximations converge with reliable speed of convergence. The aim of this project is to derive the required transformation in a systematic way and solve a number of typical problems.

