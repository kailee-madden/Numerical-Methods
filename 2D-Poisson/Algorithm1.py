"""
METHOD/ALGORITHM:
Uses algorithm 1 from lecture on Feb 9, to solve the 2D Poisson equation:
    1) Initialization of A (M by M) and F (size M) where M in the total number of vertices
    2) Assembly
        for every triangle in our triangulation 
            we compute the local stiffness matrix and add it to the whole stiffness matrix
            we do the same for the load vector
    3) Restricting A and F on the interior nodes (N is the number of interior nodes)
    4) Solve U = A \ F, in Python this is U = numpy.linalg.solve(A, F)
In order to actually implement this algorithm, we import the functions completed previously
(import FiniteElementMesh and import LocalStiffnessMatrix).
Also, note that I_i^k is the global index of the ith vertex of triangle k.
So we access I_i^k by getting elementmatrix[k][i].

In addition, we compute the difference between the exact solution and the approximate.
This is done by an a approximation of the integral as defined in the problem.

INPUTS/OUTPUTS:
The input that changes is n. From n we know the number of triangles to construct
and the number of vertices which is how we know the size of our matrices.
The output is technically just the calculated difference, but could be easily
modified to actually output the Uh approximation or U exact, since we are calculating
those already.

KEY OBSERVATIONS:
The construction of the mesh matters and all previous observations from problems
1 and 2 apply. The way in which we decide to construct the mesh and stiffness matrix
(since consistency is what matters but can do different ways as long as consistent)
changes the methodology with which we compute and solve U = A \ F.
Also if mesh is shaped regularly then the error for h norm is 1st order convergence
and the error for L2 norm is 2nd order convergence.
"""

import math
import numpy
import matplotlib.pyplot as plt
import FiniteElementMesh
import LocalStiffnessMatrix

def computeU(n):
    #initialize all our relevant variables
    nodematrix = FiniteElementMesh.getNodeMatrix(n)
    elementmatrix = FiniteElementMesh.getElementMatrix(n)
    bdnodevector = FiniteElementMesh.getBDNodeVector(nodematrix, n)
    vertices = (n+1)*(n+1)
    triangles = 2*(n**2)
    A = [[[] for y in range(vertices)] for x in range(vertices)] #size M by M
    F = [0 for x in range(vertices)] #size M

    #Assembly
    for k in range(triangles):
        #compute local stiffness matrix and load vector
        #use our elementmatrix and nodematrix to create the map of global to local indices
        v0 = elementmatrix[k][0] #gives us the vertex, nodematrix gives us the coordinates of this vertex
        v1 = elementmatrix[k][1]
        v2 = elementmatrix[k][2]
        localstiffnessmatrix = LocalStiffnessMatrix.localStiffnessMatrix(nodematrix[v0], nodematrix[v1], nodematrix[v2])
        localloadvector = LocalStiffnessMatrix.localLoadVector(nodematrix[v0], nodematrix[v1], nodematrix[v2])
        #get correct global indices from local (Ii^k is global index of ith vertex of triangle k)
        for i in range(3):
            Iik = elementmatrix[k][i]
            for j in range(3):
                Ijk = elementmatrix[k][j]

                #add to appropriate matrix/vector
                A[Iik][Ijk][0] += localstiffnessmatrix[i][j][0]
                A[Iik][Ijk][1] += localstiffnessmatrix[i][j][1]
                F[Ijk] += localloadvector[j]

    #Restriction to interior nodes
    restrictedA = A
    restrictedF = F
    for i in range(vertices):
        for j in range(vertices):
            #check if either of the indices is on the boundary (1 means true means on boundary)
            if bdnodevector[i] == 1 or bdnodevector[j] == 1: 
                #since not interior we delete this entry
                try:
                    del(restrictedA[i][j])
                except:
                    continue
        #since F is a vector we only want one loop (not two!) for it
        #check if the index is on the boundary
        if bdnodevector[i] == 1:
            try:
                del(restrictedF[i])
            except:
                continue
    
    #Solve the linear system of equations
    F_array = numpy.array(restrictedF)
    A_array = numpy.array(restrictedA)
    U = numpy.linalg.solve(A_array, F_array)

    return 

def exactU(x1, x2): #this is given to us
    output = ((2*(math.pi)**2)**-1)*(math.sin(math.pi * x1))*(math.sin(math.pi * x2)) + ((5*(math.pi)**2)**-1)*(math.sin(math.pi * x1))*(math.sin(2*math.pi * x2))
    return output

def exactUSolution(h):
    #create exact U solution
    U = []
    x = 0
    y = 0
    while x <= 1:
        while y <= 1:
            U.append(exactU(x,y))
            y += h
        x += h
    U_array = numpy.array(U)
    return U_array

def difference(exactU, approxU, v0, v1, v2, n):
    #we will calculate this difference for each triangle K
    B = [[] for x in range(2)]
    B[0] = [v1[0]-v0[0], v1[1]-v0[1]]
    B[1] = [v2[0]-v0[0], v2[1]-v0[0]]
    B_array = numpy.array(B)
    det_B = numpy.linalg.det(B)
    K = abs(det_B)/2.0 #this is an approximation formula for the area of the triangle K (from course lecture)

    #calculate the m_i vector which is dependent on our vertices (is midpoint between two vertices)
    if i ==2:
        m_i = [(v2[0]+v0[0])/2.0, (v2[1]+v0[1])/2.0]
    elif i == 1:
        m_i = [(v1[0]+v2[0])/2.0, (v1[1]+v2[1])/2.0]
    else:
        m_i = [(v0[0]+v1[0])/2.0, (v0[1]+v1[1])/2.0]

    triangles = 2*(n**2)
    diffVector = [0 for x in range(triangles)]
    #compute the difference for each triangle k
    for k in range(triangles):
        #internal summation for v(mi)
        summation = 0
        for i in range(3):
            summation += numpy.subtract((exactU[i]-approxU[i])**2) #this is v(x) as defined by (u(x)-uh(x)^2)
        #for each triangle k we approximate the integral of v(x)dx on k by the difference vector
        diffVector.append(K * (1.0/3.0) * summation)

    return diffVector



if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement

    #Our testing setup
    N = [4, 8, 16, 32]
    triangles = 2*(n**2)

    for n in N:
        approxU = computeU(n)
        exactU = exactUSolution(1.0/n)
        diff = []
        for k in triangles:
            v0 = elementmatrix[k][0] 
            v1 = elementmatrix[k][1]
            v2 = elementmatrix[k][2]
            diffVector = difference(exactU, approxU, v0, v1, v2)
            diff.append(diffVector)
        print(diff)