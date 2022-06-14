"""
METHOD/ALGORITHM:
Using reference triangles, we can construct phi(j) to get an a(ij)^k for each of our triangles k.
Similarly we can use reference triangles and take the middle point m(i) to use the quadrature
rule to approximate our local load vector f.

INPUTS/OUTPUTS:
Our input is a given triangle K with three vertices v1, v2, and v3. Note that this input
can be derived from the construction of our finite element mesh (hence how problems 1 and 2 for 
the homework will be bound together in problem 3). 
An additional input is the function f(x1, x2) that we are using the finite element method on. 
Our outputs are the local stiffness matrix and the local load vector. These two values
only are representing the specific triangle we gave as input, but can be used in a loop
to count all the triangles and construct the full stiffness matrix and load vector.

KEY OBSERVATIONS:
The way in which we construct phi and phi hat matters and can be the difference
between correct and incorrect results.
The approximation for the area of K only works because of the abs() around det_B,
since otherwise the value could be negative which does not make sense for an area.
The same B is being used for the local stiffness matrix and the local load vector, 
so a future optimization could be to calculate that outside of the functions, however,
this would limit the functionality of the functions to run completely independently.
"""

import numpy
import math

def localStiffnessMatrix(v0, v1, v2):
    #initialize all the relevant variables first
    localstiffnessmatrix = [[0 for y in range(3)] for x in range(3)] #need it to be 3 by 3 since i,j = 1,2,3

    #get everything dependent on B
    #B = (v1-v0, v2-v0) a 2 by 2 matrix
    B = [[] for x in range(2)]
    B[0] = [v1[0]-v0[0], v1[1]-v0[1]]
    B[1] = [v2[0]-v0[0], v2[1]-v0[0]]
    B_array = numpy.array(B)
    B_transpose = numpy.transpose(B_array)
    try:
        B_T_inverse = numpy.linalg.inv(B_transpose)
    except:
        B_T_inverse = [[0,0],[0,0]]
    det_B = numpy.linalg.det(B)

    #get our phi related variable
    #phi_hat_1 = (-1, -1)^T, phi_hat_2= (1,0)^T, phi_hat_3 = (0,1)^T
    phi = [[-1,-1], [1,0], [0,1]]

    #compute the whole local stiffness matrix
    #formula is (1/2)(B^T^-1 * phi_hat(j)) * (B^T^-1 * phi_hat(i)) * det(B) for i,j = 1,2,3
    for i in range(3):
        phi_hat_i = numpy.transpose(numpy.array(phi[i]))
        for j in range(3):
            phi_hat_j = numpy.transpose(numpy.array(phi[j]))
            localstiffnessmatrix[i][j] = .5*(numpy.matmul(B_T_inverse, phi_hat_j)) *(numpy.matmul(B_T_inverse, phi_hat_i)) *det_B

    return localstiffnessmatrix

def localLoadVector(v0, v1, v2):
    localloadvector = [0 for x in range(3)]

    #get everything dependent on B
    #B = (v1-v0, v2-v0) a 2 by 2 matrix
    B = [[] for x in range(2)]
    B[0] = [v1[0]-v0[0], v1[1]-v0[1]]
    B[1] = [v2[0]-v0[0], v2[1]-v0[0]]
    B_array = numpy.array(B)
    det_B = numpy.linalg.det(B)
    K = abs(det_B)/2.0 #this is an approximation formula for the area of the triangle K (from course lecture)

    for j in range(3):
        summation = 0
        for i in range(3):
            #calculate the phi_j(m_i) - this will always be either 0 or 1/2 depending on m_i's position relative to phi_j
            if j == 0 and i ==1:
                phi_j = 0
            elif j==1 and i ==2:
                phi_j = 0
            elif j==2 and i ==0:
                phi_j = 0
            else:
                phi_j = .5
            #calculate the m_i vector which is dependent on our vertices (is midpoint between two vertices)
            if i ==2:
                m_i = [(v2[0]+v0[0])/2.0, (v2[1]+v0[1])/2.0]
            elif i == 1:
                m_i = [(v1[0]+v2[0])/2.0, (v1[1]+v2[1])/2.0]
            else:
                m_i = [(v0[0]+v1[0])/2.0, (v0[1]+v1[1])/2.0]
            summation += function_f(m_i[0], m_i[1]) * phi_j
            
        localloadvector[j] = K * (1.0/3.0) * summation

    return localloadvector

def function_f(x1, x2):
    output = (math.sin(math.pi * x1))*(math.sin(math.pi * x2)) + (math.sin(math.pi * x1))*(math.sin(2*math.pi * x2))
    return output

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement

    #Our testing setup
    v0 = [0.0, 0.0]
    v1 = [0.5, 0.5]
    v2 = [0.0, 1.0]

    #function calls 
    localstiffnessmatrix = localStiffnessMatrix(v0, v1, v2)
    localloadvector = localLoadVector(v0, v1, v2)


    #print out specific lines to test
    print("7th row of node matrix: {}".format(localstiffnessmatrix))
    print("Local Load Vector: {}".format(localloadvector))