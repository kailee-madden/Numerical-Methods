import math
import numpy
import matplotlib.pyplot as plt

"""our exact ODE solution to -u'+u = 2x
with initial conditions u(0)=u(1)=0"""
def u(x):
    B = 2/(((math.e)**1)-((math.e)**-1))
    A = -B
    return A*(math.e)**x + B*(math.e)**-x + 2*x

"""F function as defined in the problem"""
def f(x):
    return 2*x

"""Run the Finite Difference Method - assuming u(0)=u(1)=0
Returns the max abs error value"""
def finite_difference_method(h, f_func, u_func):
    #create A, our interval based on h, and our x values
    A = A_creation(h)
    intervals = int(1/h)
    X = [round(num*h, 1) for num in range(intervals+1)]

    #create our F matrix/array based on the initial conditions where boundary points are 0 and the f function
    F = [f(x) for x in X]
    F[0] = 0
    F[-1] = 0
    F_array = numpy.array(F)
    #solve the linear system of equations AU=F for U approx
    U = numpy.linalg.solve(A, F_array)
    #create exact U solution
    U_exact = [u(x) for x in X]
    U_exact_array = numpy.array(U_exact)
    #get max absolute error
    mae = numpy.abs(numpy.subtract(U, U_exact_array)).max()
    return mae

"""Create the tridiagonal A matrix"""
def A_creation(h):
    intervals = int(1/h)
    #list comprehension to construct matrix of 0s
    A = [[0 for a in range(intervals+1)] for a in range(intervals+1)]
    #calculate the diagonal elements, and divide by h^2
    main_diag = round((2 + (h**2))/((h**2)), 1)
    off_diag = round((-1/(h**2)), 1)
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
    MAE = []
    #repeat finite difference for all H values
    for h in H:
        #call finite difference method and pass h, function f, and function u
        MAE.append(finite_difference_method(h, f, u))

    #Plotting
    plt.plot(H, MAE)
    plt.title("Finite Difference Method plot of Max Absolute Errors")
    plt.xlabel("H values (our inputs)")
    plt.ylabel("Max Absolute Error between U_h and U")
    plt.scatter(H,MAE,s=300,color='purple',zorder=2)
    plt.grid()
    plt.savefig("Finite Difference Method.png")