import math
import numpy
import matplotlib.pyplot as plt

"""our exact solution"""
def u(x1, x2):
    output = ((2*(math.pi)**2)**-1)*(math.sin(math.pi * x1))*(math.sin(math.pi * x2)) + ((5*(math.pi)**2)**-1)*(math.sin(math.pi * x1))*(math.sin(2*math.pi * x2))
    return output

"""F function as defined in the problem"""
def f(x1, x2):
    output = (math.sin(math.pi * x1))*(math.sin(math.pi * x2)) + (math.sin(math.pi * x1))*(math.sin(2*math.pi * x2))
    return output

"""Run the Finite Difference Method 
Returns the max abs error value"""
def finite_difference_method(h, f_func, u_func):
    #create A, our interval based on h, and our x values
    A = A_creation(h)
    intervals = int(1/h)
    X = [round(num*h, 1) for num in range(intervals+1)]

    #create our F matrix/array based on the initial conditions where boundary points are 0 and the f function
    F = []
    for x in range(intervals+1):
        for y in range(intervals+1):
            F.append(f_func(x,y))
    F_array = numpy.array(F)
    #solve the linear system of equations AU=F for U approx
    U = numpy.linalg.solve(A, F_array)
    #create exact U solution
    U_exact = []
    for x in range(intervals+1):
        for y in range(intervals+1):
            U_exact.append(u_func(x,y))
    U_exact_array = numpy.array(U_exact)
    #get max absolute error
    mae = numpy.abs(numpy.subtract(U, U_exact_array)).max()
    return mae

"""Create the block tridiagonal A matrix made up of I and A-old matrices
where A-old is tridiagonal and I is identity matrix"""
def A_creation(h):
    intervals = int(1/h)
    #list comprehension to construct matrix of 0s (n-1)^2 by (n-1)^2
    A = [[0 for a in range((intervals+1)**2)] for a in range((intervals+1)**2)]
    #calculate the diagonal elements, and divide by h^2
    main_diag = round((4/(h**2)), 1)
    off_diag = round((-1/(h**2)), 1)
    #set the correct diagonals and off diagonals value
    for i in range((intervals+1)**2):
        for j in range((intervals+1)**2):
            #main diagonal
            if i==j:
                A[i][j] = main_diag
            #usual off diagonals
            elif i+1 ==j or i-1 == j:
                A[i][j] = off_diag
            #our block I off diagonals
            elif i+intervals==j or i-intervals ==j:
                A[i][j] = off_diag
    #make a numpy matrix
    A_matrix = numpy.matrix(A)
    return A_matrix

#Main function to call other functions and do the solving process for all the necessary values of h
if __name__ == "__main__":
    H = [1/10, 1/20, 1/40, 1/80]
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