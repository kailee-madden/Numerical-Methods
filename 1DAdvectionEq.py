""" 
METHOD/ALGORITHM:
We have four algorithms that we are using to produce an A matrix from which we can using the linear system of equations
U^(n+1) = A U^n to solve for our approximate solution. We can then compare this to the exact U solution.
Our four algorithms are the FTCS scheme, upwind scheme, Lax-Friedrichs scheme and Lax-Wendroff scheme.
FTCS (forward in time, central in space: Uj,n+1 = Uj,n -(ak/2h)(Uj+1,n - Uj-1,n)
Upwind: We are using a<0 so the wave propagates towards the left and the right side of j is called the upwind side (note: 
we could've done the opposite and all the observations would remain the same just the A matrix would be slightly different).
Since a<0 then Uj,n+1 = Uj,n + (ak/h)(Uj+1,n -Uj,n)
Lax-Friedrichs: Uj,n+1 = .5(1-(ak/h))Uj+1,n + .5(1+(ak/h))Uj-1,n
Lax-Wendroff:
In addition, we are using both smooth and nonsmooth initial conditions. These initial conditions change both our approximations
and our exact solutions. Thus for each of our above schemes, we also run the scheme with each of the initial conditions (smooth
and nonsmooth). So there are 8 total variations in our methodology.

INPUTS/OUTPUTS:
There are a number of changing inputs: the scheme we are using, the initial conditions (smooth vs not smooth), 
the step size, and we could customize this further as well, but for the sake of simplicity for these particular tests, 
these were the inputs that changed. In addition, I made it so that we could modify the equation easily and use the same
functions (ex. can change the "a" parameter in the 1D advection eq.)
The outputs are the actual exact U solution for smooth and nonsmooth, the approximate U solution for our chosen scheme for
smooth and nonsmooth, and the error between these two solutions.

KEY OBSERVATIONS:
Using different testing parameters (ex. changing the value of h which is our step size), we can see the following observations
each of the numerical schemes using smooth and nonsmooth initial conditions:
FTCS: first order method in space and time, unstable (but can be conditionally stable with extremely small time step relative to space step),
as we move to a smaller k value, you can see the propagation of the wave/shifting, for nonsmooth the errors are more extreme
Upwind: stable if k <= h/abs(a) (the CFL condition) this condition is satisfied in the second iteration with k=.05, this iteration
shows a shifting of the wave but also more clearly shows the wave itself, for nonsmooth the errors are more extreme, first order in 
space and time
Lax-Friedrichs: conditionally stable according to the CFL condition with k <= h/abs(a), similar to FTCS but with a dissipation term of 1/2,
first order accuracy
Lax-Wendroff: conditionally stable according to the CFL condition, causes oscillations near the corners suggesting dispersion effect, 
more accurate than some previous schemes, second order in space and time
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


""" Exact smooth, nonsmooth and initial conditions smooth and nonsmooth """
def exactSmoothU(x, t): #our smooth exact solution
    output = math.sin(2*math.pi * (x-t))
    return output

def exactNonsmoothU(x, t): #our non-smooth exact solution
    num = x-t
    if num <0.5:
        return 1
    else:
        return 0 

def exactUSolution(h,T, smooth):
    #create exact U solution
    dim = int(1/h)
    U = np.zeros(dim+1)
    if smooth:
        for i in range(0, dim+1):
            U[i] = exactSmoothU(i*h, T) 
    else:
        for i in range(0, dim+1):
            U[i] = exactNonsmoothU(i*h, T) 
    return U

def initialConditionFunction(x, smooth):
    if smooth: #use our smooth initial conditions
        output = math.sin(2*math.pi*x)
    else: #use our nonsmooth initial conditions
        if x < 0.5:
            output = 1
        else:
            output = 0
    return output


""" All of our different schemes to use to approximately solve for U """
def FTCS(h,k,a, smooth, N):
    # A is 1 on diagonal and -ak/2h on above and below off diagonals

    #create A matrix
    dim = int(1/h)
    A= np.zeros((dim+1, dim+1))
    A[dim, dim] = 1
    A[dim, dim-1] = (-a*k)/(2.0*h)
    A[0, 0] = 1
    A[0, 1] = (-a*k)/(2.0*h)
    for i in range(1, dim):
        A[i,i] = 1
        A[i, i+1] = (-a*k)/(2.0*h)
        A[i, i-1] = (-a*k)/(2.0*h)

    U = [initialConditionFunction(j*h, smooth) for j in range(dim+1)] #create U^0 vector

    for n in range(1,N): #the number of iterations we want to do
        #solve for U^n+1
        U_old = U
        U = np.linalg.solve(A, U_old)

    return U

def upwind(h, k, a, smooth, N):
    # A is 1-(ak/h) on diagonal and ak/h on diagonal+1 (this is because we chose a<0)
    #create A matrix
    dim = int(1/h)
    A= np.zeros((dim+1, dim+1))
    A[dim, dim] = 1-(a*k)/(2.0*h)
    for i in range(0, dim):
        A[i,i] = 1-(a*k)/(2.0*h)
        A[i, i+1] = (-a*k)/(2.0*h)

    U = [initialConditionFunction(j*h, smooth) for j in range(dim+1)] #create U^0 vector

    for n in range(1,N): #the number of iterations we want to do
        #solve for U^n+1
        U_old = U
        U = np.linalg.solve(A, U_old)

    return U

def LaxFriedrichs(h, k, a, smooth, N):
    # A is 0 on diagonal, (1+ak/h)/2 below diagonal, and (1-ak/h)/2 above diagonal
    #create A matrix
    dim = int(1/h)
    A= np.zeros((dim+1, dim+1))
    for i in range(0, dim+1):
        for j in range(0, dim+1):
            if i == j:
                continue
            elif i > j: #below diagonal
                A[i][j] = (1+(a*k/h))/2.0
            else: #above diagonal
                A[i][j] = (1-(a*k/h))/2.0

    U = [initialConditionFunction(j*h, smooth) for j in range(dim+1)] #create U^0 vector

    for n in range(1,N): #the number of iterations we want to do
        #solve for U^n+1
        U_old = U
        U = np.linalg.solve(A, U_old)

    return U

def LaxWendroff(h, k, a, smooth, N):
    # A is 1-(ak/h)^2 on the diagonal, ((ak/h)^2)/2 + (ak/h)/2 on the +1 diagonal, ((ak/h)^2)/2 - (ak/h)/2 on the -1 diagonal
    #create A matrix
    dim = int(1/h)
    r = a*k*1.0/h
    A= np.zeros((dim+1, dim+1))
    A[0, 0] = 1.0- r**2
    A[0, 1] = (r**2)/2.0 - r/2.0
    A[0, dim] = (r**2)/2.0 + r/2.0
    A[dim, 0] = (r**2)/2.0 - r/2.0
    A[dim, dim-1] = (r**2)/2.0 + r/2.0
    A[dim, dim] = 1.0- r**2
    for i in range(1, dim):
        A[i,i] = 1.0- r**2
        A[i, i+1] = (r**2)/2.0 - r/2.0
        A[i, i-1] = (r**2)/2.0 + r/2.0

    U = [initialConditionFunction(j*h, smooth) for j in range(dim+1)] #create U^0 vector

    for n in range(1,N): #the number of iterations we want to do
        #solve for U^n+1
        U_old = U
        U = np.linalg.solve(A, U_old)

    return U

if __name__ == "__main__": #uncomment based on which plot you want to generate
    h = 0.1

    # U_exact = exactUSolution(h, 0, True) # change to True or False depending on whether you want smooth or nonsmooth solution
    # plt.plot(U_exact)
    # plt.title("Exact Smooth U Solution with T = 0")
    # plt.savefig("Exact Smooth U 1DAdvection.png")

    #U_FTCS = FTCS(h, .1, 1, True, 10) # change to True or False depending on whether you want smooth or nonsmooth solution
    # U_FTCS = FTCS(h, .05, 1, False, 20)
    # plt.plot(U_FTCS)
    # plt.title("FTCS Approx Solution, 20 iterations, k=.05, h=.1, Smooth Initial Conditions")
    # plt.savefig("FTCS Smooth 2")
    # plt.title("FTCS Approx Solution, 20 iterations, k=.05, h=.1, NonSmooth Initial Conditions")
    # plt.savefig("FTCS NonSmooth 2")

    #U_upwind = upwind(h, .1, 1, False, 10) # change to True or False depending on whether you want smooth or nonsmooth solution
    # U_upwind = upwind(h, .05, 1, True, 20)
    # plt.plot(U_upwind)
    # plt.title("Upwind Approx, 10 iterations, k=.1, h=.1, NonSmooth")
    # plt.savefig("Upwind NonSmooth")
    # plt.title("Upwind Approx, 20 iterations, k=.05, h=.1, Smooth")
    # plt.savefig("Upwind Smooth 2")


    #U_LaxFriedrichs = LaxFriedrichs(h, .0005, 1, False, 10) # change to True or False depending on whether you want smooth or nonsmooth solution
    # U_LaxFriedrichs = LaxFriedrichs(h, .005, 1, False, 20)
    # plt.plot(U_LaxFriedrichs)
    # plt.title("LF, 10 iterations, k=.005, h=.1, NonSmooth")
    # plt.savefig("LF NonSmooth")
    # plt.title("LF, 20 iterations, k=.0005, h=.1, NonSmooth")
    # plt.savefig("LF NonSmooth 2")


    #U_LaxWendroff = LaxWendroff(h, .0005, 1, True, 20) # change to True or False depending on whether you want smooth or nonsmooth solution
    # U_LaxWendroff = LaxWendroff(h, .005, 1, False, 10)
    # plt.plot(U_LaxWendroff)
    # plt.title("LW, 10 iterations, k=.005, h=.1, Smooth")
    # plt.savefig("LW Smooth")
    # plt.title("LW, 20 iterations, k=.0005, h=.1, Smooth")
    # plt.savefig("LW Smooth 2")





