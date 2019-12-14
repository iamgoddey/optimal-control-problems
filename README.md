# optimal-control-problems
Numerical Methods for Optimal Control Problems:

The solution to optimal control problems for ordinary differential equations can be obtained by applying Pontryagin's minimum principle. This usually yields in general a non-linear boundary value problem which has to be solved numerically. Such methods are called indirect methods as they are solving first order necessary conditions. The boundary value problems to be solved are of the form \[ y'= f(t,y,u), \quad 0\le t \le t_f \] with the set of boundary conditions expressed in the form  \[g_1(y(0)) = 0 \mbox{ and } g_2(y(t_f)) = 0\] where $ y $ is the vector function of the state and co-state variables, $ u $ is the scalar or vector function of control variables, $ g_1 \in \RR^n \mbox{ and } g_2 \in \RR^m $ for some values of $ m \mbox{ and } n \mbox{ with } 1<m<n$ where each vector functions $ g_1 \mbox{ and } g_2$ are independent. The boundary value problem also requires the satisfaction of two-point or multi-point boundary conditions. Of special interest are optimal control problems with constraints either for the control or the state variables. For such problems, the right hand side of the differential equation may be piecewise smooth, that is, there are points at which the right hand $ f(t,y,u) $ jumps as the control variable $ u $ may show discontinuities.

The non-smooth behaviour of the right hand side inhibits a reliable convergence of the numerical approximations towards the exact solution. A way around this problem is a transformation of the points with non-smooth behaviour to known, fixed locations $ t_i \in [0,t_f] $. Then it is obvious that the numerical approximations converge with reliable speed of convergence. The aim of this project is to derive the required transformation in a systematic way and solve a number of typical problems.

