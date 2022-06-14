""" 
METHOD/ALGORITHM:
The two algorithms used here at crank nicolson and explicit euler. For the functions that compute these approximate solutions,
the algorithm is commented below the function. We create the A and B matrices based on our inputs k and h and then use the initial
U to iteratively refine our U vector. The error calculations are done by subtracting the actual U and the approximate U and then
finding the absolute max value in that vector.
When doing the error calculations, we iteratively change h and k to make them get closer to 0. Since h determines the size of our
A and U and k determines the iterations for creating U, as h and k get smaller, the calculation to approximate U gets slower (since
it is using larger and larger matrices and running more iterations).

INPUTS/OUTPUTS:
The inputs that change are our fixed time T, our k (time steps), and h (space steps). These inputs change the size of the matrices
and vectors we are working with and also change the number of iterations that we perform to approximate our U vector.
The outputs are the approximate U vectors from each method, the error when subtracting the actual U and the approximate U, and 
the slopes that show the error changing over time as our h and k go to 0, as well as the chart for the actual U.

KEY OBSERVATIONS:
For crank nicolson, by fixing time at T=1 and running iterations to determine the error, we can see that as h and k go to 0,
with a fixed relation between h and k (k=4h), that the method is second order accurate. Thus the smaller h and k we pick the 
more accurate the approximation will be (but also the slower it will be to calculate).
For explicit euler, by fixing time at T=1 and running iterations to determine the error, we can see that as h and k go to 0, 
with a fixed relation between h and k (k = .4h^2), that the method is also second order accurate. However, compared to crank
nicolson, there are more time steps required to be performed (hence relation of k=.4h^2). And again, the finer the grid, the slower the approximation will
be to calculate.
For explicit euler, if we use the relation k=0.6h^2 instead, we can observe that it is unstable. That is, the error does not
converge and using k and h going to 0 does not provide a more and more accurate approximation, as we would like it to. Thus,
we can conclude that the relation between k and h has signification impact on the accuracy of our approximation.
Note: for second order convergence, that means the slope should be 2 (so that is what we look for in the output graphs.)
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def exactU(x, t): #this is given to us
    pi_sq = (math.pi)**2
    e_part_1 = (math.e)**(-pi_sq * t)
    e_part_2 = (math.e)**(-9*pi_sq * t)
    sin_1 = math.sin(math.pi *x)
    sin_2 = math.sin(math.pi * x * 3)
    part_1 = sin_1 * e_part_1
    part_2 = sin_2 * e_part_2
    output = part_1 - part_2

    return output

def exactUSolution(h,T):
    #create exact U solution
    dim = int(1/h)
    U = np.zeros(dim+1)
    for i in range(0, dim+1):
        U[i] = exactU(i*h, T) 

    return U

def crankNicolson(h, k, N):
    #(1+k/h^2)Uj,n+1 - (k/2h^2)Uj+1,n+1 - (k/2h^2)Uj-1,n+1 = (1-k/h^2)Uj,n + (k/2h^2)Uj+1,n + (k/2h^2)Uj-1,n

    #create A matrix
    dim = int(1/h)
    A= np.zeros((dim+1, dim+1))
    A[0, 0] = 1+(k/(h**2))
    A[0, 1] = -k/(2*(h**2))
    A[dim, dim-1] = -k/(2*(h**2))
    A[dim, dim] = 1+(k/(h**2))

    for i in range(1, dim):
        A[i,i] = 1+(k/(h**2))
        A[i, i+1] = -k/(2*(h**2))
        A[i, i-1] = -k/(2*(h**2))
    A_inv = np.linalg.inv(A)
    #create B matrix
    B= np.zeros((dim+1, dim+1))
    B[0, 0] = 1-(k/(h**2))
    B[0, 1] = k/(2*(h**2))
    B[dim, dim-1] = k/(2*(h**2))
    B[dim, dim] = 1-(k/(h**2))
    for i in range(1, dim):
        B[i,i] = 1-(k/(h**2))
        B[i, i+1] = k/(2*(h**2))
        B[i, i-1] = k/(2*(h**2))
   # print("B matrix: {}".format(B)) 
    U = [initialConditionFunction(j*h) for j in range(dim+1)] #create U^0 vector  
    
    for n in range(1,N): #number of iterations
        #solve for U^n+1
        U_old = U
        AB = np.matmul(A_inv, B)
        U = np.linalg.solve(AB, U_old)

    return U

def explicitEuler(h,k,N):
    #Uj,n+1 = Uj,n + k/h^2 (Uj+1,n + Uj-1,n -2Uj,n)

    #create A matrix
    dim = int(1/h)
    A= np.zeros((dim+1, dim+1))
    A[0, 0] = 1-(2.0*k/(h**2))
    A[0, 1] = k/(h**2)
    A[dim, dim-1] = k/(h**2)
    A[dim, dim] = 1-(2.0*k/(h**2))
    for i in range(1, dim):
        A[i,i] = 1-(2.0*k/(h**2))
        A[i, i+1] = k/(h**2)
        A[i, i-1] = k/(h**2)
   # print("A matrix: {}".format(A))

    U = [initialConditionFunction(j*h) for j in range(dim+1)] #create U^0 vector

    for n in range(1,N): #the number of iterations we want to do
        #solve for U^n+1
        U_old = U
       # print("A matrix again {}".format(A))
        #print("U old {}".format(U_old))
        U = np.linalg.solve(A, U_old)

    return U

def initialConditionFunction(x):
    output = math.sin(math.pi * x) - math.sin(3* math.pi *x)
    return output

if __name__ == "__main__": #uncomment based on which plot you want to generate
    #Basic plots with T=0, h=0.025, k=0.1
#    T=0
#    h = .05
#    k = .01
   #U_euler = explicitEuler(h, k, int(T/k))
   #U_crank = crankNicolson(h, k, int(T/k))
   #U_exact = exactUSolution(h, T)
#    plt.plot(U_exact)
#    plt.savefig("Exact U.png")
#    plt.title("Exact U Solution with T = 0")
#    plt.plot(U_euler)
#    plt.title("Euler Approx with T=0, h=.05, k=.01")
#    plt.savefig("Euler U.png")
#    plt.plot(U_crank)
#    plt.title("Crank Nicolson Approx with T=0, h=.05, k=.01")
#    plt.savefig("CN U.png")

    #Problem 1: Calculate error between Crank Nicolson and Exact U at fixed time T= 1 as k and h go to zero with k=4h
    # T = 1
    # error = []
    # H = [.1, .05, .025]
    # K = [.4, .2, .01]
    # for i in range(3): #iterations of error calculations
    #     k = K[i]
    #     h = H[i]
    #     U_crank = crankNicolson(h, k, int(T/k))
    #     U_exact = exactUSolution(h, T)  #sine curve decaying
    #     error.append(np.amax(np.subtract(U_exact, U_crank))) #get max value of the error vector
    # plt.plot(H,error)
    # plt.title("Stable Crank Nicolson Error (problem 1)")
    # plt.savefig("Crank Nicolson Error.png")
    

    #Problem 2: Calculate error between Explicit Euler and Exact U at fixed time T=1 as k and h go to zero with k=.4*h^2
    # T= 1
    # H = [.2, .1, .05]
    # K = [.016, .004, .001]
    # error = []
    # for i in range(3): #iterations of error calculations
    #     k = K[i]
    #     h = H[i]
    #     U_euler = explicitEuler(h, k, int(T/k))
    #     U_exact = exactUSolution(h, T)
    #     error.append(np.amax(np.subtract(U_exact, U_euler))) #get max value of the error vector

    # plt.plot(H,error)
    # plt.title("Stable Euler Error (problem 2)")
    # plt.savefig("Euler Error.png")

    #Problem 3: Calculate error between Explicit Euler and Exact U at fixed time T=1 as k and h go to zero with k=.6*h^2
    # T= 1
    # H = [.2, .1, .05]
    # K = [.024, .006, .0015]
    # error = []
    # for i in range(3): #iterations of error calculations
    #     k = K[i]
    #     h = H[i]
    #     U_euler = explicitEuler(h, k, int(T/k))
    #     U_exact = exactUSolution(h, T)
    #     error.append(np.amax(np.subtract(U_exact, U_euler))) #get max value of the error vector
    # plt.plot(H, error)
    # plt.title("Unstable Euler Error (problem 2)")
    # plt.savefig("Unstable Euler Error.png")



