import math
import numpy
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad
#import dolfin
#from dolfin.fem.norms import norm

"""our exact ODE solution to -u'+u = 2x
with initial conditions u(0)=u(1)=0"""
def u(x):
    B = 2/(((math.e)**1)-((math.e)**-1))
    A = -B
    val = A*(math.e)**x + B*(math.e)**-x + 2*x
    return val

"""F function as defined in the problem
The approximate solution uh(x) = PN−1
j=1 ujϕj (x)
where {ϕj (x)} are the piecewise linear basis functions.
quad is a scipy function that takes the integral"""
def f(x, h):
    f_x = lambda x: (2*x)*(1/h)
    return quad(f_x, 0, 1)

"""Run the Finite Difference Method - assuming u(0)=u(1)=0
Returns the error in the H1 seminorm, the error in the L2 norm 
"""
def finite_difference_method(h, f_func, u_func):
    #create A, our interval based on h, and our x values
    A = A_creation(h)
    intervals = int(1/h)
    X = [round(num*h, 1) for num in range(intervals+1)]

    #create our F matrix/array 
    F = []
    for x in X:
        num, err = f(x,h)
        F.append(num)
    F_array = numpy.array(F)
    print(F_array)
    #solve the linear system of equations AU=F for U approx
    U = numpy.linalg.solve(A, F_array)
    #create exact U solution
    U_exact = [u(x) for x in X]
    U_exact_array = numpy.array(U_exact)
    print(U)
    print("split between")
    print(U_exact_array)
    #get our two error values
    Diff = numpy.subtract(U, U_exact_array)
    #L2_error = norm(Diff, norm_type='L2', mesh=2)
    #H1_error = norm(Diff, norm_type='H1', mesh=2)
    L2_error =  numpy.sum(numpy.power((U-U_exact_array),2))
    grad = numpy.gradient(U-U_exact_array)
    L2_grad = numpy.sum(numpy.power((grad),2))
    H1_error = (L2_error + L2_grad)**(1/2)
    
    return H1_error, L2_error

"""Create the tridiagonal A matrix"""
def A_creation(h):
    intervals = int(1/h)
    #list comprehension to construct matrix of 0s
    A = [[0 for a in range(intervals+1)] for a in range(intervals+1)]
    #calculate the diagonal elements, and divide by h^2
    main_diag = 2*(1/h)
    off_diag = -1/h
    #set the correct diagonals and off diagonals value
    for i in range(intervals+1):
        for j in range(intervals+1):
            #main diagonal
            if i==j:
                A[i][j] = main_diag
            #off diagonals
            elif i+1 ==j or i-1 == j:
                A[i][j] = off_diag
    #make a numpy matrix
    A_matrix = numpy.matrix(A)
    return A_matrix

#Main function to call other functions and do the solving process for all the necessary values of h
if __name__ == "__main__":
    H = [1/10, 1/20, 1/40, 1/80, 1/160]
    H1 = []
    L2 = []
    #repeat finite element for all H values
    for h in H:
        #call finite element method and pass h, function f, and function u
        h1, l2 = finite_difference_method(h, f, u)
        H1.append(h1)
        L2.append(l2)

    #Plotting
    plt.plot(H, H1)
    plt.title("Finite Element Method H1 Error Plot")
    plt.xlabel("H values (our inputs)")
    plt.ylabel("H1 Error between U_h and U")
    plt.scatter(H,H1,s=300,color='purple',zorder=2)
    plt.grid()
    plt.savefig("Finite Element H1.png")

    plt.plot(H, L2)
    plt.title("Finite Element Method L2 Error Plot")
    plt.xlabel("H values (our inputs)")
    plt.ylabel("L2 Error between U_h and U")
    plt.scatter(H,L2,s=300,color='purple',zorder=2)
    plt.grid()
    plt.savefig("Finite Element L2.png")