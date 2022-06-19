#include <vector>
using namespace std;
Given a matrix A of size n by n
Create matrix L of size n by n
For i from 0 to n:
	For j from 0 to n:
		Sum = 0
		If row == col: # this is for the diagonals
			For k from 0 to j:
				Sum += L[j][k] * L[j][k]
			L[j][j] = sqrt(A[j][j] - sum)
		Else:
			For k from 0 to j: # this is for the non diagonals
				Sum += L[i][k] * L[j][k]
			L[i][j] = (A[i][j] - sum) / L[j][j]