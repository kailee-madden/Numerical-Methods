"""
METHOD/ALGORITHM:
This code generates a uniform finite element mesh for omega with 2n^2 trianges for a given positive integer n
The data structure includes
    1) A node matrix of size (total # vertices)*2
    2) An element matrix of size (total # triangles)*3
    3) A bdNode vector of size (total # vertices)
Note: Python using zero-based indexing, so our vertices start at 0. 
Thus the 7th row of the node matrix is printed using 6 to index.
So based on the triangulation drawn for this problem, the exact same triangulation 
has been constructed except everything is n-1 instead of n.
So the 8th row of the element matrix is the triangle labeled 8 on the drawing
but is the triangle corresponding to row 7 in this code. The corresponding
vertices are [9, 13, 14] in the code and on the drawing they are [10, 14, 15].
The algorithm for the node matrix is just dividing our grid into intervals based on n.
The algorithm for the boundary vector is just checking whether the coordinates for a
vertex in the node matrix include a 0 or 1 (the boundary coordinates).
The algorithm for the element matrix uses two clear patterns in the triangle construction.
These patterns show the relationship between the vertices that construct a triangle (ie
how much is the difference between v1 and v2) and the relationship remains consistent for
the whole pattern (the two patterns are the first half of the triangles and the second half).
And the counter is used to identify when we jump from one row to the next row so we can 
iterate the vertices accordingly.

INPUTS/OUTPUTS:
The input that changes is n. From n we know the number of triangles to construct.
In the future, this could be further generalized to allow for different boundary
inputs, but for this case, we were asked to construct it for boundary = [0,1].
The outputs are the nodematrix, elementmatrix, and boundarynodevector.

KEY OBSERVATIONS:
This code works for a particular triangulation, however there are other ones possible.
To construct other triangulations, the code would need to be modified because the patterns
that allow for identification of vertices in the triangles would change.
The code could also be further optimized by constructing the node matrix and boundary
node vector simultaneously. However, by allowing their construction in different functions,
we provide the user more flexibility."""

def getNodeMatrix(n):
    #initialize all our necessary variables
    length = float(1.0/n)
    vertices = (n+1)*(n+1)
    nodematrix = [[] for x in range(vertices)] 
    colcoord = 0.0 
    rowcoord = 0.0
    vertex = 0

    #using our length we iterate until have gone through all our vertices
    for c in range(1, n+2):
        rowcoord = 0
        for r in range(1, n+2):
            nodematrix[vertex] = [colcoord, rowcoord]
            vertex += 1
            rowcoord = length * r
        colcoord = length * c
    return nodematrix

def getElementMatrix(n):
    #initialize all our necessary variables
    length = float(1.0/n)
    triangles = 2*(n**2)
    elementmatrix = [[] for x in range(triangles)]
    triangle = 0
    v1 = 0
    v2 = n+1
    v3 = n+2
    counter = 0

    #construct the first half of the triangles
    while triangle < triangles/2:
        elementmatrix[triangle] = [v1, v2, v3]
        triangle += 1
        if counter == n-1:
            v1 += 2
            v2 += 2
            v3 += 2
            counter = 0
        else: 
            v1 += 1
            v2 += 1
            v3 += 1
            counter += 1
    
    #reset the variables as needed (triangle stays the same)
    v1 = 0
    v2 = 1
    v3 = n+2
    counter = 0
    #construct the second half of the triangles
    while triangle < triangles:
        elementmatrix[triangle] = [v1, v2, v3]
        triangle += 1
        if counter == n-1:
            v1 += 2
            v2 += 2
            v3 += 2
            counter = 0
        else: 
            v1 += 1
            v2 += 1
            v3 += 1
            counter += 1

    return elementmatrix 

def getBDNodeVector(nodematrix, n):
    bdnodevector = [0 for node in nodematrix] #initialize the node vector to have all 0 entries
    vertices = (n+1)*(n+1)
    for vertex in range(vertices):
            if nodematrix[vertex][0] == 0.0 or nodematrix[vertex][0] == 1.0: #check if we have boundary coordinates
                bdnodevector[vertex] = 1
            elif nodematrix[vertex][1] == 0.0 or nodematrix[vertex][1] == 1.0:
                bdnodevector[vertex] = 1
    return bdnodevector




if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement

    #Our testing setup
    n = 5

    #function calls 
    nodematrix = getNodeMatrix(n)
    elementmatrix = getElementMatrix(n)
    bdnodevector = getBDNodeVector(nodematrix, n)

    #print out specific lines to test
    print("7th row of node matrix: {}".format(nodematrix[6]))
    print("8th row of element matrix: {}".format(elementmatrix[7]))
    print("12th component of bdNode vector: {}".format(bdnodevector[11]))