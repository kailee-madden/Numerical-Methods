#include <vector>
using namespace std;
for  j=1:n
  
    vj=xj
  
endfor 
for  j=1:n
  
    rjj=‖vj‖2
  
    qj=vj/rjj
  
   for  k=j+1:n
  
      rjk=qTjvk
  
      vk=vk−rjkqj
  
   endfor 
endfor