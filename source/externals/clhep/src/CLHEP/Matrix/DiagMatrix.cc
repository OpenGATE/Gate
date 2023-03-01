// -*- C++ -*-
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//

#include <iostream>
#include <string.h>
#include <cmath>

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Matrix/DiagMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Vector.h"

#ifdef HEP_DEBUG_INLINE
#include "CLHEP/Matrix/DiagMatrix.icc"
#endif

namespace CLHEP {

// Simple operation for all elements

#define SIMPLE_UOP(OPER)          \
  HepMatrix::mIter a=m.begin();            \
  HepMatrix::mIter e=m.begin()+num_size(); \
  for(;a<e; a++) (*a) OPER t;

#define SIMPLE_BOP(OPER)          \
   HepMatrix::mIter a=m.begin();            \
   HepMatrix::mcIter b=hm2.m.begin();         \
   HepMatrix::mIter e=m.begin()+num_size(); \
   for(;a<e; a++, b++) (*a) OPER (*b);

#define SIMPLE_TOP(OPER)          \
   HepMatrix::mcIter a=hm1.m.begin();            \
   HepMatrix::mcIter b=hm2.m.begin();         \
   HepMatrix::mIter t=mret.m.begin();         \
   HepMatrix::mcIter e=hm1.m.begin()+hm1.nrow; \
   for( ;a<e; a++, b++, t++) (*t) = (*a) OPER (*b);

#define CHK_DIM_2(r1,r2,c1,c2,fun) \
   if (r1!=r2 || c1!=c2)  { \
    HepGenMatrix::error("Range error in DiagMatrix function " #fun "(1)."); \
   }

#define CHK_DIM_1(c1,r2,fun) \
   if (c1!=r2) { \
    HepGenMatrix::error("Range error in DiagMatrix function " #fun "(2)."); \
   }

// static constant

#if defined(__sun) || !defined(__GNUG__)
//
// Sun CC 4.0.1 has this bug.
//
double HepDiagMatrix::zero = 0;
#else
const double HepDiagMatrix::zero = 0;
#endif

// Constructors. (Default constructors are inlined and in .icc file)

HepDiagMatrix::HepDiagMatrix(int p)
   : m(p), nrow(p)
{
}

HepDiagMatrix::HepDiagMatrix(int p, int init)
   : m(p), nrow(p)
{   
   switch(init)
   {
   case 0:
      m.assign(nrow,0);
      break;

   case 1:
      {
	 HepMatrix::mIter a=m.begin();
	 HepMatrix::mIter b=m.begin() + p;
	 for( ; a<b; a++) *a = 1.0;
	 break;
      }
   default:
      error("DiagMatrix: initialization must be either 0 or 1.");
   }
}

HepDiagMatrix::HepDiagMatrix(int p, HepRandom &r)
  : m(p), nrow(p)
{
   HepMatrix::mIter a = m.begin();
   HepMatrix::mIter b = m.begin() + num_size();
   for(;a<b;a++) *a = r();
}
//
// Destructor
//
HepDiagMatrix::~HepDiagMatrix() {
}

HepDiagMatrix::HepDiagMatrix(const HepDiagMatrix &hm1)
   : HepGenMatrix(hm1), m(hm1.nrow), nrow(hm1.nrow)
{
   m = hm1.m;
}

//
//
// Sub matrix
//
//

HepDiagMatrix HepDiagMatrix::sub(int min_row, int max_row) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(max_row-min_row+1);
{
#else
{
  HepDiagMatrix mret(max_row-min_row+1);
#endif
  if(max_row > num_row())
    error("HepDiagMatrix::sub: Index out of range");
  HepMatrix::mIter a = mret.m.begin();
  HepMatrix::mcIter b = m.begin() + min_row - 1;
  HepMatrix::mIter e = mret.m.begin() + mret.num_row();
  for(;a<e;) *(a++) = *(b++);
  return mret;
}

HepDiagMatrix HepDiagMatrix::sub(int min_row, int max_row)
{
  HepDiagMatrix mret(max_row-min_row+1);
  if(max_row > num_row())
    error("HepDiagMatrix::sub: Index out of range");
  HepMatrix::mIter a = mret.m.begin();
  HepMatrix::mIter b = m.begin() + min_row - 1;
  HepMatrix::mIter e = mret.m.begin() + mret.num_row();
  for(;a<e;) *(a++) = *(b++);
  return mret;
}

void HepDiagMatrix::sub(int row,const HepDiagMatrix &hm1)
{
  if(row <1 || row+hm1.num_row()-1 > num_row() )
    error("HepDiagMatrix::sub: Index out of range");
  HepMatrix::mcIter a = hm1.m.begin();
  HepMatrix::mIter b = m.begin() + row - 1;
  HepMatrix::mcIter e = hm1.m.begin() + hm1.num_row();
  for(;a<e;) *(b++) = *(a++);
}

//
// Direct sum of two matricies
//

HepDiagMatrix dsum(const HepDiagMatrix &hm1,
				     const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
  return mret(hm1.num_row() + hm2.num_row(), 0);
{
#else
{
  HepDiagMatrix mret(hm1.num_row() + hm2.num_row(),
				       0);
#endif
  mret.sub(1,hm1);
  mret.sub(hm1.num_row()+1,hm2);
  return mret;
}

HepDiagMatrix HepDiagMatrix::operator- () const 
#ifdef HEP_GNU_OPTIMIZED_RETURN
      return hm2(nrow);
{
#else
{
   HepDiagMatrix hm2(nrow);
#endif
   HepMatrix::mcIter a=m.begin();
   HepMatrix::mIter b=hm2.m.begin();
   HepMatrix::mcIter e=m.begin()+num_size();
   for(;a<e; a++, b++) (*b) = -(*a);
   return hm2;
}

   

HepMatrix operator+(const HepMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),
			 hm1.num_col(),hm2.num_col(),+);
  mret += hm2;
  return mret;
}

HepMatrix operator+(const HepDiagMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm2);
{
#else
{
  HepMatrix mret(hm2);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),
			 hm1.num_col(),hm2.num_col(),+);
  mret += hm1;
  return mret;
}

HepDiagMatrix operator+(const HepDiagMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.nrow);
{
#else
{
  HepDiagMatrix mret(hm1.nrow);
#endif
  CHK_DIM_1(hm1.nrow,hm2.nrow,+);
  SIMPLE_TOP(+)
  return mret;
}

HepSymMatrix operator+(const HepDiagMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm2);
{
#else
{
  HepSymMatrix mret(hm2);
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),+);
  mret += hm1;
  return mret;
}

HepSymMatrix operator+(const HepSymMatrix &hm2,const HepDiagMatrix &hm1)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm2);
{
#else
{
  HepSymMatrix mret(hm2);
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),+);
  mret += hm1;
  return mret;
}

//
// operator -
//

HepMatrix operator-(const HepMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),
			 hm1.num_col(),hm2.num_col(),-);
  mret -= hm2;
  return mret;
}
HepMatrix operator-(const HepDiagMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),
			 hm1.num_col(),hm2.num_col(),-);
  mret -= hm2;
  return mret;
}

HepDiagMatrix operator-(const HepDiagMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.nrow);
{
#else
{
  HepDiagMatrix mret(hm1.nrow);
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),-);
  SIMPLE_TOP(-)
  return mret;
}
HepSymMatrix operator-(const HepDiagMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepSymMatrix mret(hm1);
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),-);
  mret -= hm2;
  return mret;
}

HepSymMatrix operator-(const HepSymMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepSymMatrix mret(hm1);
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),-);
  mret -= hm2;
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This file contains
   The two argument functions *,/. They call copy constructor and then /=,*=.
   Also contains v_times_vT(const HepVector &v).
   ----------------------------------------------------------------------- */

HepDiagMatrix operator/(
const HepDiagMatrix &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepDiagMatrix mret(hm1);
#endif
  mret /= t;
  return mret;
}

HepDiagMatrix operator*(const HepDiagMatrix &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepDiagMatrix mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepDiagMatrix operator*(double t,const HepDiagMatrix &hm1)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepDiagMatrix mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepMatrix operator*(const HepMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(),hm2.num_col());
{
#else
  {
    HepMatrix mret(hm1.num_row(),hm2.num_col());
#endif
    CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
    HepMatrix::mcIter mit1=hm1.m.begin();
    HepMatrix::mIter mir=mret.m.begin();
    for(int irow=1;irow<=hm1.num_row();irow++) {
      HepMatrix::mcIter mcc = hm2.m.begin();
      for(int icol=1;icol<=hm1.num_col();icol++) {
	*(mir++) = *(mit1++) * (*(mcc++));
      }
    }
    return mret;
  }

HepMatrix operator*(const HepDiagMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(),hm2.num_col());
{
#else
{
  HepMatrix mret(hm1.num_row(),hm2.num_col());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  HepMatrix::mcIter mit1=hm2.m.begin();
  HepMatrix::mIter mir=mret.m.begin();
  HepMatrix::mcIter mrr = hm1.m.begin();
  for(int irow=1;irow<=hm2.num_row();irow++) {
    for(int icol=1;icol<=hm2.num_col();icol++) {
      *(mir++) = *(mit1++) * (*mrr);
    }
    mrr++;
  }
  return mret;
}

HepDiagMatrix operator*(const HepDiagMatrix &hm1,const HepDiagMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepDiagMatrix mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  HepMatrix::mIter a = mret.m.begin();
  HepMatrix::mcIter b = hm1.m.begin();
  HepMatrix::mcIter c = hm2.m.begin();
  HepMatrix::mIter e = mret.m.begin() + hm1.num_col();
  for(;a<e;) *(a++) = *(b++) * (*(c++));
  return mret;
}

HepVector operator*(const HepDiagMatrix &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepVector mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  HepGenMatrix::mIter mir=mret.m.begin();
  HepGenMatrix::mcIter mi1 = hm1.m.begin(), mi2 = hm2.m.begin();
  for(int icol=1;icol<=hm1.num_col();icol++) {
    *(mir++) = *(mi1++) * *(mi2++);
  }
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains the assignment and inplace operators =,+=,-=,*=,/=.
   ----------------------------------------------------------------------- */

HepMatrix & HepMatrix::operator+=(const HepDiagMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  int n = num_row();
  mIter mrr = m.begin();
  HepMatrix::mcIter mr = hm2.m.begin();
  for(int r=1;r<=n;r++) {
    *mrr += *(mr++);
    if(r<n) mrr += (n+1);
  }
  return (*this);
}

HepSymMatrix & HepSymMatrix::operator+=(const HepDiagMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  HepMatrix::mIter a=m.begin();
  HepMatrix::mcIter b=hm2.m.begin();
  for(int i=1;i<=num_row();i++) {
    *a += *(b++);
    if(i<num_row()) a += (i+1);
  }
  return (*this);
}

HepDiagMatrix & HepDiagMatrix::operator+=(const HepDiagMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

HepMatrix & HepMatrix::operator-=(const HepDiagMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),-=);
  int n = num_row();
  mIter mrr = m.begin();
  HepMatrix::mcIter mr = hm2.m.begin();
  for(int r=1;r<=n;r++) {
    *mrr -= *(mr++);
    if(r<n) mrr += (n+1);
  }
  return (*this);
}

HepSymMatrix & HepSymMatrix::operator-=(const HepDiagMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  HepMatrix::mIter a=m.begin();
  HepMatrix::mcIter b=hm2.m.begin();
  for(int i=1;i<=num_row();i++) {
    *a -= *(b++);
    if(i<num_row()) a += (i+1);
  }
  return (*this);
}

HepDiagMatrix & HepDiagMatrix::operator-=(const HepDiagMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

HepDiagMatrix & HepDiagMatrix::operator/=(double t)
{
  SIMPLE_UOP(/=)
  return (*this);
}

HepDiagMatrix & HepDiagMatrix::operator*=(double t)
{
  SIMPLE_UOP(*=)
  return (*this);
}

HepMatrix & HepMatrix::operator=(const HepDiagMatrix &hm1)
{
   if(hm1.nrow*hm1.nrow != size_)
   {
      size_ = hm1.nrow * hm1.nrow;
      m.resize(size_);
   }
   nrow = hm1.nrow;
   ncol = hm1.nrow;
   int n = nrow;
   m.assign(size_,0); 
   mIter mrr = m.begin();
   HepMatrix::mcIter mr = hm1.m.begin();
   for(int r=1;r<=n;r++) {
      *mrr = *(mr++);
      if(r<n) mrr += (n+1);
   }
   return (*this);
}

HepDiagMatrix & HepDiagMatrix::operator=(const HepDiagMatrix &hm1)
{
   if(hm1.nrow != nrow)
   {
      nrow = hm1.nrow;
      m.resize(nrow);
   }
   m=hm1.m;
   return (*this);
}

// Print the Matrix.

std::ostream& operator<<(std::ostream &os, const HepDiagMatrix &q)
{
  os << "\n";
/* Fixed format needs 3 extra characters for field, while scientific needs 7 */
  long width;
  if(os.flags() & std::ios::fixed)
    width = os.precision()+3;
  else
    width = os.precision()+7;
  for(int irow = 1; irow<= q.num_row(); irow++)
    {
      for(int icol = 1; icol <= q.num_col(); icol++)
	{
	  os.width(width);
	  os << q(irow,icol) << " ";
	}
      os << std::endl;
    }
  return os;
}

HepDiagMatrix HepDiagMatrix::
apply(double (*f)(double, int, int)) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(num_row());
{
#else
{
  HepDiagMatrix mret(num_row());
#endif
  HepMatrix::mcIter a = m.begin();
  HepMatrix::mIter b = mret.m.begin();
  for(int ir=1;ir<=num_row();ir++) {
    *(b++) = (*f)(*(a++), ir, ir);
  }
  return mret;
}

void HepDiagMatrix::assign (const HepMatrix &hm1)
{
   if(hm1.num_row()!=nrow)
   {
      nrow = hm1.num_row();
      m.resize(nrow);
   }
   HepMatrix::mcIter a = hm1.m.begin();
   HepMatrix::mIter b = m.begin();
   for(int r=1;r<=nrow;r++) {
      *(b++) = *a;
      if(r<nrow) a += (nrow+1);
   }
}

void HepDiagMatrix::assign(const HepSymMatrix &hm1)
{
   if(hm1.num_row()!=nrow)
   {
      nrow = hm1.num_row();
      m.resize(nrow);
   }
   HepMatrix::mcIter a = hm1.m.begin();
   HepMatrix::mIter b = m.begin();
   for(int r=1;r<=nrow;r++) {
      *(b++) = *a;
      if(r<nrow) a += (r+1);
   }
}

HepSymMatrix HepDiagMatrix::similarity(const HepMatrix &hm1) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepSymMatrix mret(hm1.num_row());
#endif
  CHK_DIM_1(num_row(),hm1.num_col(),"similarity");
//  HepMatrix temp = hm1*(*this);
// If hm1*(*this) has correct dimensions, then so will the hm1.T multiplication.
// So there is no need to check dimensions again.
  HepMatrix::mIter mrc = mret.m.begin();
  for(int r=1;r<=mret.num_row();r++) {
    HepMatrix::mcIter mrr = hm1.m.begin()+(r-1)*hm1.num_col();
    HepMatrix::mcIter mc = hm1.m.begin();
    for(int c=1;c<=r;c++) {
      HepMatrix::mcIter mi = m.begin();
      double tmp = 0;
      HepMatrix::mcIter mr = mrr;
      for(int i=0;i<hm1.num_col();i++)
	tmp+=*(mr++) * *(mc++) * *(mi++);
      *(mrc++) = tmp;
    }
  }
  return mret;
}

double HepDiagMatrix::similarity(const HepVector &hm1) const
{
  double mret;
  CHK_DIM_1(num_row(),hm1.num_row(),similarity);
  HepMatrix::mcIter mi = m.begin();
  HepMatrix::mcIter mv = hm1.m.begin();
  mret = *(mv)* *(mv)* *(mi++);
  mv++;
  for(int i=2;i<=hm1.num_row();i++) {
    mret+=*(mv)* *(mv)* *(mi++);
    mv++;
  }
  return mret;
}

HepSymMatrix HepDiagMatrix::similarityT(const HepMatrix &hm1) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_col());
{
#else
{
  HepSymMatrix mret(hm1.num_col());
#endif
  CHK_DIM_1(num_col(),hm1.num_row(),similarityT);
//  Matrix temp = (*this)*hm1;
// If hm1*(*this) has correct dimensions, then so will the hm1.T multiplication.
// So there is no need to check dimensions again.
  for(int r=1;r<=mret.num_row();r++)
    for(int c=1;c<=r;c++)
      {
	HepMatrix::mcIter mi = m.begin();
	double tmp = hm1(1,r)*hm1(1,c)* *(mi++);
	for(int i=2;i<=hm1.num_row();i++)
	  tmp+=hm1(i,r)*hm1(i,c)* *(mi++);
	mret.fast(r,c) = tmp;
      }
  return mret;
}

void HepDiagMatrix::invert(int &ierr) {
  int n = num_row();
  ierr = 1;
  HepMatrix::mIter hmm = m.begin();
  int i;
  for(i=0;i<n;i++) {
    if(*(hmm++)==0) return;
  }
  ierr = 0;
  hmm = m.begin();
  for(i=0;i<n;i++) {
    *hmm = 1.0 / *hmm;
    hmm++;
  }  
}

double HepDiagMatrix::determinant() const {
   double d = 1.0;
   HepMatrix::mcIter end = m.begin() + nrow;
   for (HepMatrix::mcIter p=m.begin(); p < end; p++)
      d *= *p;
   return d;
}

double HepDiagMatrix::trace() const {
   double d = 0.0;
   HepMatrix::mcIter end = m.begin() + nrow;
   for (HepMatrix::mcIter p=m.begin(); p < end; p++)
      d += *p;
   return d;
}

}  // namespace CLHEP
