/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCompressedVoxel_H
#define GateCompressedVoxel_H 1

#include <vector>
#include <valarray>
#include <set>
#include <algorithm>
#include <iostream>

// ---------------------------------

typedef unsigned short int usi;

/* 

   Note : x1 is the slowest varying index (z) and x3 the fastest (x)

   Variable size voxel at position x1, x2, x3 with size dx1, dx2, dx3

   index:               0   1   2   3    4    5    6
   variable "a":      { x1, x2, x3, dx1, dx2, dx3, value }

*/

class GateCompressedVoxel{
public:
  GateCompressedVoxel(usi x1=0, usi x2=0, usi x3=0, usi dx1=0, usi dx2=0, usi dx3=0, usi value=0){
    a[0]=x1; a[1]=x2; a[2]=x3; a[3]=dx1; a[4]=dx2; a[5]=dx3; a[6]=value;
  }
  
  // Compare this voxel with rhs on specified indices only (contained in valarray l)
  bool compare(const GateCompressedVoxel& rhs, const std::valarray<unsigned short>& l)const{
    bool ans(true);
    for(unsigned int i=0; i<l.size(); i++) ans = ans && a[l[i]]==rhs.a[l[i]] ;
    return ans ;
  }

  // Compute the position difference between this voxel and rhs
  std::valarray<unsigned short int> positionDifference(const GateCompressedVoxel& rhs){
    std::valarray<unsigned short int> positionA(a,3);
    std::valarray<unsigned short int> positionB(rhs.a,3);
    positionA -= positionB;
    return positionA;
  }
  
  // array like accessors
  inline       usi& operator[] (int n)     { return a[n] ; }
  inline const usi& operator[] (int n)const{ return a[n] ; }
  

private:
  usi a[7];
};

typedef std::vector<GateCompressedVoxel> voxelSet;
std::ostream& operator << (std::ostream& , const GateCompressedVoxel&) ;


// ---------------------------------

// parameterizable relational operator used by sort

class GateCompressedVoxelOrdering{
public:
  //  Sort order idx1: major;  idx3: minor
  GateCompressedVoxelOrdering(int i0, int i1, int i2){
    index[0]=i0; index[1]=i1; index[2]=i2; 
  }
  
  bool operator() (const GateCompressedVoxel& lhs,  const GateCompressedVoxel& rhs){ 
    register int i;
    for( i=0; i<3; i++){
      if ( lhs[index[i]] < rhs[index[i]] ) return true;
      else 
	if ( lhs[index[i]] > rhs[index[i]] ) return false;
    }
    
    return false;		// Covers the lhs==rhs case (strict ordering)
  }
  
private:
  int index[3];
  
};




#endif
