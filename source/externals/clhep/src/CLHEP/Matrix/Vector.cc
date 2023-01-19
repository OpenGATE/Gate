// -*- C++ -*-
// ---------------------------------------------------------------------------
//

#include <iostream>
#include <string.h>

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Utility/thread_local.h"

#ifdef HEP_DEBUG_INLINE
#include "CLHEP/Matrix/Vector.icc"
#endif

namespace CLHEP {

// Simple operation for all elements

#define SIMPLE_UOP(OPER)          \
   HepGenMatrix::mIter a=m.begin();            \
   HepGenMatrix::mIter e=m.begin()+num_size(); \
   for(;a<e; a++) (*a) OPER t;

#define SIMPLE_BOP(OPER)          \
   mIter a=m.begin();            \
   mcIter b=hm2.m.begin();               \
   mcIter e=m.begin()+num_size(); \
   for(;a<e; a++, b++) (*a) OPER (*b);

#define SIMPLE_TOP(OPER)          \
   HepGenMatrix::mcIter a=hm1.m.begin();            \
   HepGenMatrix::mcIter b=hm2.m.begin();         \
   HepGenMatrix::mIter t=mret.m.begin();         \
   HepGenMatrix::mcIter e=hm1.m.begin()+hm1.num_size(); \
   for( ;a<e; a++, b++, t++) (*t) = (*a) OPER (*b);

#define CHK_DIM_2(r1,r2,c1,c2,fun) \
   if (r1!=r2 || c1!=c2)  { \
     HepGenMatrix::error("Range error in Vector function " #fun "(1)."); \
   }

#define CHK_DIM_1(c1,r2,fun) \
   if (c1!=r2) { \
     HepGenMatrix::error("Range error in Vector function " #fun "(2)."); \
   }

// Constructors. (Default constructors are inlined and in .icc file)

HepVector::HepVector(int p)
   : m(p), nrow(p)
{
}

HepVector::HepVector(int p, int init)
   : m(p), nrow(p)
{
   switch (init)
   {
   case 0:
      m.assign(p,0);
      break;
      
   case 1:
      {
	 mIter e = m.begin() + nrow;
	 for (mIter i=m.begin(); i<e; i++) *i = 1.0;
	 break;
      }
      
   default:
      error("Vector: initialization must be either 0 or 1.");
   }
}

HepVector::HepVector(int p, HepRandom &r)
   : m(p), nrow(p)
{
   HepGenMatrix::mIter a = m.begin();
   HepGenMatrix::mIter b = m.begin() + nrow;
   for(;a<b;a++) *a = r();
}


//
// Destructor
//
HepVector::~HepVector() {
}

HepVector::HepVector(const HepVector &hm1)
   : HepGenMatrix(hm1), m(hm1.nrow), nrow(hm1.nrow)
{
   m = hm1.m;
}

//
// Copy constructor from the class of other precision
//


HepVector::HepVector(const HepMatrix &hm1)
   : m(hm1.nrow), nrow(hm1.nrow)
{
   if (hm1.num_col() != 1)
      error("Vector::Vector(Matrix) : Matrix is not Nx1");
   
   m = hm1.m;
}

// trivial methods

int HepVector::num_row() const {return nrow;} 
int HepVector::num_size() const {return nrow;} 
int HepVector::num_col() const { return 1; }

// operator()

#ifdef MATRIX_BOUND_CHECK
double & HepVector::operator()(int row, int col)
{
  if( col!=1 || row<1 || row>nrow)
     error("Range error in HepVector::operator(i,j)");
#else
double & HepVector::operator()(int row, int)
{
#endif

  return *(m.begin()+(row-1));
}

#ifdef MATRIX_BOUND_CHECK
const double & HepVector::operator()(int row, int col) const 
{
  if( col!=1 || row<1 || row>nrow)
     error("Range error in HepVector::operator(i,j)");
#else
const double & HepVector::operator()(int row, int) const 
{
#endif

  return *(m.begin()+(row-1));
}

// Sub matrix

HepVector HepVector::sub(int min_row, int max_row) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return vret(max_row-min_row+1);
{
#else
{
  HepVector vret(max_row-min_row+1);
#endif
  if(max_row > num_row())
    error("HepVector::sub: Index out of range");
  HepGenMatrix::mIter a = vret.m.begin();
  HepGenMatrix::mcIter b = m.begin() + min_row - 1;
  HepGenMatrix::mIter e = vret.m.begin() + vret.num_row();
  for(;a<e;) *(a++) = *(b++);
  return vret;
}

HepVector HepVector::sub(int min_row, int max_row)
{
  HepVector vret(max_row-min_row+1);
  if(max_row > num_row())
    error("HepVector::sub: Index out of range");
  HepGenMatrix::mIter a = vret.m.begin();
  HepGenMatrix::mIter b = m.begin() + min_row - 1;
  HepGenMatrix::mIter e = vret.m.begin() + vret.num_row();
  for(;a<e;) *(a++) = *(b++);
  return vret;
}

void HepVector::sub(int row,const HepVector &v1)
{
  if(row <1 || row+v1.num_row()-1 > num_row())
    error("HepVector::sub: Index out of range");
  HepGenMatrix::mcIter a = v1.m.begin();
  HepGenMatrix::mIter b = m.begin() + row - 1;
  HepGenMatrix::mcIter e = v1.m.begin() + v1.num_row();
  for(;a<e;) *(b++) = *(a++);
}

//
// Direct sum of two matricies
//

HepVector dsum(const HepVector &hm1,
				     const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
  return mret(hm1.num_row() + hm2.num_row(), 0);
{
#else
{
  HepVector mret(hm1.num_row() + hm2.num_row(),
				       0);
#endif
  mret.sub(1,hm1);
  mret.sub(hm1.num_row()+1,hm2);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This section contains
   The two argument functions +,-. They call the copy constructor and +=,-=.
   ----------------------------------------------------------------------- */
HepVector HepVector::operator- () const 
#ifdef HEP_GNU_OPTIMIZED_RETURN
      return hm2(nrow);
{
#else
{
   HepVector hm2(nrow);
#endif
   HepGenMatrix::mcIter a=m.begin();
   HepGenMatrix::mIter b=hm2.m.begin();
   HepGenMatrix::mcIter e=m.begin()+num_size();
   for(;a<e; a++, b++) (*b) = -(*a);
   return hm2;
}

   

HepVector operator+(const HepMatrix &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm2);
{
#else
{
  HepVector mret(hm2);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),hm1.num_col(),1,+);
  mret += hm1;
  return mret;
}

HepVector operator+(const HepVector &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepVector mret(hm1);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),1,hm2.num_col(),+);
  mret += hm2;
  return mret;
}

HepVector operator+(const HepVector &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepVector mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),+);
  SIMPLE_TOP(+)
  return mret;
}

//
// operator -
//

HepVector operator-(const HepMatrix &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret;
{
#else
{
  HepVector mret;
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),hm1.num_col(),1,-);
  mret = hm1;
  mret -= hm2;
  return mret;
}

HepVector operator-(const HepVector &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepVector mret(hm1);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),1,hm2.num_col(),-);
  mret -= hm2;
  return mret;
}

HepVector operator-(const HepVector &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepVector mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),-);
  SIMPLE_TOP(-)
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This file contains
   The two argument functions *,/. They call copy constructor and then /=,*=.
   ----------------------------------------------------------------------- */

HepVector operator/(
const HepVector &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepVector mret(hm1);
#endif
  mret /= t;
  return mret;
}

HepVector operator*(const HepVector &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepVector mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepVector operator*(double t,const HepVector &hm1)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepVector mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepVector operator*(const HepMatrix &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepVector mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  HepGenMatrix::mcIter hm1p,hm2p,vp;
  HepGenMatrix::mIter m3p;
  double temp;
  m3p=mret.m.begin();
  for(hm1p=hm1.m.begin();hm1p<hm1.m.begin()+hm1.num_row()*hm1.num_col();hm1p=hm2p)
    {
      temp=0;
      vp=hm2.m.begin();
      hm2p=hm1p;
      while(hm2p<hm1p+hm1.num_col())
	temp+=(*(hm2p++))*(*(vp++));
      *(m3p++)=temp;
    }
  return mret;
}

HepMatrix operator*(const HepVector &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(),hm2.num_col());
{
#else
{
  HepMatrix mret(hm1.num_row(),hm2.num_col());
#endif
  CHK_DIM_1(1,hm2.num_row(),*);
  HepGenMatrix::mcIter hm1p;
  HepMatrix::mcIter hm2p;
  HepMatrix::mIter mrp=mret.m.begin();
  for(hm1p=hm1.m.begin();hm1p<hm1.m.begin()+hm1.num_row();hm1p++)
    for(hm2p=hm2.m.begin();hm2p<hm2.m.begin()+hm2.num_col();hm2p++)
      *(mrp++)=*hm1p*(*hm2p);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains the assignment and inplace operators =,+=,-=,*=,/=.
   ----------------------------------------------------------------------- */

HepMatrix & HepMatrix::operator+=(const HepVector &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),1,+=);
  SIMPLE_BOP(+=)
  return (*this);
}

HepVector & HepVector::operator+=(const HepMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),1,hm2.num_col(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

HepVector & HepVector::operator+=(const HepVector &hm2)
{
  CHK_DIM_1(num_row(),hm2.num_row(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

HepMatrix &  HepMatrix::operator-=(const HepVector &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),1,-=);
  SIMPLE_BOP(-=)
  return (*this);
}

HepVector & HepVector::operator-=(const HepMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),1,hm2.num_col(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

HepVector & HepVector::operator-=(const HepVector &hm2)
{
  CHK_DIM_1(num_row(),hm2.num_row(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

HepVector & HepVector::operator/=(double t)
{
  SIMPLE_UOP(/=)
  return (*this);
}

HepVector & HepVector::operator*=(double t)
{
  SIMPLE_UOP(*=)
  return (*this);
}

HepMatrix & HepMatrix::operator=(const HepVector &hm1)
{
   if(hm1.nrow != size_)
   {
      size_ = hm1.nrow;
      m.resize(size_);
   }
   nrow = hm1.nrow;
   ncol = 1;
   m = hm1.m;
   return (*this);
}

HepVector & HepVector::operator=(const HepVector &hm1)
{
   if(hm1.nrow != nrow)
   {
      nrow = hm1.nrow;
      m.resize(nrow);
   }
   m = hm1.m;
   return (*this);
}

HepVector & HepVector::operator=(const HepMatrix &hm1)
{
   if (hm1.num_col() != 1)
      error("Vector::operator=(Matrix) : Matrix is not Nx1");
   
   if(hm1.nrow != nrow)
   {
      nrow = hm1.nrow;
      m.resize(nrow);
   }
   m = hm1.m;
   return (*this);
}

HepVector & HepVector::operator=(const Hep3Vector &v)
{
   if(nrow != 3)
   {
      nrow = 3;
      m.resize(nrow);
   }
   m[0] = v.x();
   m[1] = v.y();
   m[2] = v.z();
   return (*this);
}

//
// Copy constructor from the class of other precision
//


// Print the Matrix.

std::ostream& operator<<(std::ostream &os, const HepVector &q)
{
  os << std::endl;
/* Fixed format needs 3 extra characters for field, while scientific needs 7 */
  long width;
  if(os.flags() & std::ios::fixed)
    width = os.precision()+3;
  else
    width = os.precision()+7;
  for(int irow = 1; irow<= q.num_row(); irow++)
    {
      os.width(width);
      os << q(irow) << std::endl;
    }
  return os;
}

HepMatrix HepVector::T() const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(1,num_row());
{
#else
{
  HepMatrix mret(1,num_row());
#endif
  mret.m = m;
  return mret;
}

double dot(const HepVector &v1,const HepVector &v2)
{
  if(v1.num_row()!=v2.num_row())
     HepGenMatrix::error("v1 and v2 need to be the same size in dot(HepVector, HepVector)");
  double d= 0;
  HepGenMatrix::mcIter a = v1.m.begin();
  HepGenMatrix::mcIter b = v2.m.begin();
  HepGenMatrix::mcIter e = a + v1.num_size();
  for(;a<e;) d += (*(a++)) * (*(b++));
  return d;
}

HepVector HepVector::
apply(double (*f)(double, int)) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(num_row());
{
#else
{
  HepVector mret(num_row());
#endif
  HepGenMatrix::mcIter a = m.begin();
  HepGenMatrix::mIter b = mret.m.begin();
  for(int ir=1;ir<=num_row();ir++) {
    *(b++) = (*f)(*(a++), ir);
  }
  return mret;
}

void HepVector::invert(int &) {
   error("HepVector::invert: You cannot invert a Vector");
}

HepVector solve(const HepMatrix &a, const HepVector &v)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return vret(v);
{
#else
{
  HepVector vret(v);
#endif
  static CLHEP_THREAD_LOCAL int max_array = 20;
  static CLHEP_THREAD_LOCAL int *ir = new int [max_array+1];

  if(a.ncol != a.nrow)
     HepGenMatrix::error("Matrix::solve Matrix is not NxN");
  if(a.ncol != v.nrow)
     HepGenMatrix::error("Matrix::solve Vector has wrong number of rows");

  int n = a.ncol;
  if (n > max_array) {
    delete [] ir;
    max_array = n;
    ir = new int [max_array+1];
  }
  double det;
  HepMatrix mt(a);
  int i = mt.dfact_matrix(det, ir);
  if (i!=0) {
    for (i=1;i<=n;i++) vret(i) = 0;
    return vret;
  }
  double s21, s22;
  int nxch = ir[n];
  if (nxch!=0) {
    for (int hmm=1;hmm<=nxch;hmm++) {
      int ij = ir[hmm];
      i = ij >> 12;
      int j = ij%4096;
      double te = vret(i);
      vret(i) = vret(j);
      vret(j) = te;
    }
  }
  vret(1) = mt(1,1) * vret(1);
  if (n!=1) {
    for (i=2;i<=n;i++) {
      s21 = -vret(i);
      for (int j=1;j<i;j++) {
	s21 += mt(i,j) * vret(j); 
      }
      vret(i) = -mt(i,i)*s21;
    }
    for (i=1;i<n;i++) {
      int nmi = n-i;
      s22 = -vret(nmi);
      for (int j=1;j<=i;j++) {
	s22 += mt(nmi,n-j+1) * vret(n-j+1);
      }
      vret(nmi) = -s22;
    }
  }
  return vret;
}

}  // namespace CLHEP
