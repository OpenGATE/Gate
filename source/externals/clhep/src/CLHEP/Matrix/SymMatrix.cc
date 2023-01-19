// -*- C++ -*-
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//

#include <iostream>
#include <string.h>
#include <float.h>        // for DBL_EPSILON

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/DiagMatrix.h"
#include "CLHEP/Matrix/Vector.h"

#ifdef HEP_DEBUG_INLINE
#include "CLHEP/Matrix/SymMatrix.icc"
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
   HepMatrix::mcIter e=m.begin()+num_size(); \
   for(;a<e; a++, b++) (*a) OPER (*b);

#define SIMPLE_TOP(OPER)          \
   HepMatrix::mcIter a=hm1.m.begin();           \
   HepMatrix::mcIter b=hm2.m.begin();         \
   HepMatrix::mIter t=mret.m.begin();         \
   HepMatrix::mcIter e=hm1.m.begin()+hm1.num_size(); \
   for( ;a<e; a++, b++, t++) (*t) = (*a) OPER (*b);

#define CHK_DIM_2(r1,r2,c1,c2,fun) \
   if (r1!=r2 || c1!=c2)  { \
     HepGenMatrix::error("Range error in SymMatrix function " #fun "(1)."); \
   }

#define CHK_DIM_1(c1,r2,fun) \
   if (c1!=r2) { \
     HepGenMatrix::error("Range error in SymMatrix function " #fun "(2)."); \
   }

// Constructors. (Default constructors are inlined and in .icc file)

HepSymMatrix::HepSymMatrix(int p)
   : m(p*(p+1)/2), nrow(p)
{
   size_ = nrow * (nrow+1) / 2;
   m.assign(size_,0);
}

HepSymMatrix::HepSymMatrix(int p, int init)
   : m(p*(p+1)/2), nrow(p)
{
   size_ = nrow * (nrow+1) / 2;

   m.assign(size_,0);
   switch(init)
   {
   case 0:
      break;
      
   case 1:
      {
	 HepMatrix::mIter a; 
	 for(int i=0;i<nrow;++i) {
            a = m.begin() + (i+1)*i/2 + i;
	    *a = 1.0;
	 }
	 break;
      }
   default:
      error("SymMatrix: initialization must be either 0 or 1.");
   }
}

HepSymMatrix::HepSymMatrix(int p, HepRandom &r)
   : m(p*(p+1)/2), nrow(p)
{
   size_ = nrow * (nrow+1) / 2;
   HepMatrix::mIter a = m.begin();
   HepMatrix::mIter b = m.begin() + size_;
   for(;a<b;a++) *a = r();
}

//
// Destructor
//
HepSymMatrix::~HepSymMatrix() {
}

HepSymMatrix::HepSymMatrix(const HepSymMatrix &hm1)
   : HepGenMatrix(hm1), m(hm1.size_), nrow(hm1.nrow), size_(hm1.size_)
{
   m = hm1.m;
}

HepSymMatrix::HepSymMatrix(const HepDiagMatrix &hm1)
   : m(hm1.nrow*(hm1.nrow+1)/2), nrow(hm1.nrow)
{
   size_ = nrow * (nrow+1) / 2;

   int n = num_row();
   m.assign(size_,0);

   HepMatrix::mIter mrr = m.begin();
   HepMatrix::mcIter mr = hm1.m.begin();
   for(int r=1;r<=n;r++) {
      *mrr = *(mr++);
      if(r<n) mrr += (r+1);
   }
}

//
//
// Sub matrix
//
//

HepSymMatrix HepSymMatrix::sub(int min_row, int max_row) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(max_row-min_row+1);
{
#else
{
  HepSymMatrix mret(max_row-min_row+1);
#endif
  if(max_row > num_row())
    error("HepSymMatrix::sub: Index out of range");
  HepMatrix::mIter a = mret.m.begin();
  HepMatrix::mcIter b1 = m.begin() + (min_row+2)*(min_row-1)/2;
  int rowsize=mret.num_row();
  for(int irow=1; irow<=rowsize; irow++) {
    HepMatrix::mcIter b = b1;
    for(int icol=0; icol<irow; ++icol) {
      *(a++) = *(b++);
    }
    if(irow<rowsize) b1 += irow+min_row-1;
  }
  return mret;
}

HepSymMatrix HepSymMatrix::sub(int min_row, int max_row) 
{
  HepSymMatrix mret(max_row-min_row+1);
  if(max_row > num_row())
    error("HepSymMatrix::sub: Index out of range");
  HepMatrix::mIter a = mret.m.begin();
  HepMatrix::mIter b1 = m.begin() + (min_row+2)*(min_row-1)/2;
  int rowsize=mret.num_row();
  for(int irow=1; irow<=rowsize; irow++) {
    HepMatrix::mIter b = b1;
    for(int icol=0; icol<irow; ++icol) {
      *(a++) = *(b++);
    }
    if(irow<rowsize) b1 += irow+min_row-1;
  }
  return mret;
}

void HepSymMatrix::sub(int row,const HepSymMatrix &hm1)
{
  if(row <1 || row+hm1.num_row()-1 > num_row() )
    error("HepSymMatrix::sub: Index out of range");
  HepMatrix::mcIter a = hm1.m.begin();
  HepMatrix::mIter b1 = m.begin() + (row+2)*(row-1)/2;
  int rowsize=hm1.num_row();
  for(int irow=1; irow<=rowsize; ++irow) {
    HepMatrix::mIter b = b1;
    for(int icol=0; icol<irow; ++icol) {
      *(b++) = *(a++);
    }
    if(irow<rowsize) b1 += irow+row-1;
  }
}

//
// Direct sum of two matricies
//

HepSymMatrix dsum(const HepSymMatrix &hm1,
				     const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
  return mret(hm1.num_row() + hm2.num_row(), 0);
{
#else
{
  HepSymMatrix mret(hm1.num_row() + hm2.num_row(),
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
HepSymMatrix HepSymMatrix::operator- () const 
#ifdef HEP_GNU_OPTIMIZED_RETURN
      return hm2(nrow);
{
#else
{
   HepSymMatrix hm2(nrow);
#endif
   HepMatrix::mcIter a=m.begin();
   HepMatrix::mIter b=hm2.m.begin();
   HepMatrix::mcIter e=m.begin()+num_size();
   for(;a<e; a++, b++) (*b) = -(*a);
   return hm2;
}

   

HepMatrix operator+(const HepMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(), hm1.num_col(),hm2.num_col(),+);
  mret += hm2;
  return mret;
}
HepMatrix operator+(const HepSymMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm2);
{
#else
{
  HepMatrix mret(hm2);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),hm1.num_col(),hm2.num_col(),+);
  mret += hm1;
  return mret;
}

HepSymMatrix operator+(const HepSymMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.nrow);
{
#else
{
  HepSymMatrix mret(hm1.nrow);
#endif
  CHK_DIM_1(hm1.nrow, hm2.nrow,+);
  SIMPLE_TOP(+)
  return mret;
}

//
// operator -
//

HepMatrix operator-(const HepMatrix &hm1,const HepSymMatrix &hm2)
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
HepMatrix operator-(const HepSymMatrix &hm1,const HepMatrix &hm2)
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

HepSymMatrix operator-(const HepSymMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepSymMatrix mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_row(),hm2.num_row(),-);
  SIMPLE_TOP(-)
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This file contains
   The two argument functions *,/. They call copy constructor and then /=,*=.
   ----------------------------------------------------------------------- */

HepSymMatrix operator/(
const HepSymMatrix &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepSymMatrix mret(hm1);
#endif
  mret /= t;
  return mret;
}

HepSymMatrix operator*(const HepSymMatrix &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepSymMatrix mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepSymMatrix operator*(double t,const HepSymMatrix &hm1)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepSymMatrix mret(hm1);
#endif
  mret *= t;
  return mret;
}


HepMatrix operator*(const HepMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(),hm2.num_col());
{
#else
  {
    HepMatrix mret(hm1.num_row(),hm2.num_col());
#endif
    CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
    HepMatrix::mcIter mit1, mit2, sp,snp; //mit2=0
    double temp;
    HepMatrix::mIter mir=mret.m.begin();
    for(mit1=hm1.m.begin();
        mit1<hm1.m.begin()+hm1.num_row()*hm1.num_col();
	mit1 = mit2)
    {
      snp=hm2.m.begin();
      for(int step=1;step<=hm2.num_row();++step)
	{
	  mit2=mit1;
	  sp=snp;
	  snp+=step;
	  temp=0;
	  while(sp<snp)
	    temp+=*(sp++)*(*(mit2++));
          if( step<hm2.num_row() ) {	// only if we aren't on the last row
	    sp+=step-1;
	    for(int stept=step+1;stept<=hm2.num_row();stept++)
	      {
		temp+=*sp*(*(mit2++));
		if(stept<hm2.num_row()) sp+=stept;
	      }
	    }	// if(step
	  *(mir++)=temp;
	}	// for(step
      }	// for(mit1
    return mret;
  }

HepMatrix operator*(const HepSymMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(),hm2.num_col());
{
#else
{
  HepMatrix mret(hm1.num_row(),hm2.num_col());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  int step,stept;
  HepMatrix::mcIter mit1,mit2,sp,snp;
  double temp;
  HepMatrix::mIter mir=mret.m.begin();
  for(step=1,snp=hm1.m.begin();step<=hm1.num_row();snp+=step++)
    for(mit1=hm2.m.begin();mit1<hm2.m.begin()+hm2.num_col();mit1++)
      {
	mit2=mit1;
	sp=snp;
	temp=0;
	while(sp<snp+step)
	  {
	    temp+=*mit2*(*(sp++));
	    if( hm2.num_size()-(mit2-hm2.m.begin())>hm2.num_col() ){
	      mit2+=hm2.num_col();
	    }
	  }
        if(step<hm1.num_row()) {	// only if we aren't on the last row
	  sp+=step-1;
	  for(stept=step+1;stept<=hm1.num_row();stept++)
	    {
	      temp+=*mit2*(*sp);
	      if(stept<hm1.num_row()) {
		mit2+=hm2.num_col();
		sp+=stept;
	      }
	    }
          }	// if(step
	*(mir++)=temp;
      }	// for(mit1
  return mret;
}

HepMatrix operator*(const HepSymMatrix &hm1,const HepSymMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(),hm1.num_row());
{
#else
{
  HepMatrix mret(hm1.num_row(),hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  int step1,stept1,step2,stept2;
  HepMatrix::mcIter snp1,sp1,snp2,sp2;
  double temp;
  HepMatrix::mIter mr = mret.m.begin();
  snp1=hm1.m.begin();
  for(step1=1;step1<=hm1.num_row();++step1) {
    snp2=hm2.m.begin();
    for(step2=1;step2<=hm2.num_row();++step2)
      {
	sp1=snp1;
	sp2=snp2;
	snp2+=step2;
	temp=0;
	if(step1<step2)
	  {
	    while(sp1<snp1+step1) {
	      temp+=(*(sp1++))*(*(sp2++));
	      }
	    sp1+=step1-1;
	    for(stept1=step1+1;stept1!=step2+1;++stept1) {
	      temp+=(*sp1)*(*(sp2++));
	      if(stept1<hm2.num_row()) sp1+=stept1;
	      }
            if(step2<hm2.num_row()) {	// only if we aren't on the last row
	      sp2+=step2-1;
	      for(stept2=step2+1;stept2<=hm2.num_row();stept1++,stept2++) {
		temp+=(*sp1)*(*sp2);
		if(stept2<hm2.num_row()) {
	           sp1+=stept1;
		   sp2+=stept2;
		   }
		}	// for(stept2
	      }	// if(step2
	  }	// step1<step2
	else
	  {
	    while(sp2<snp2) {
	      temp+=(*(sp1++))*(*(sp2++));
	      }
	    if(step2<hm2.num_row()) {	// only if we aren't on the last row
	      sp2+=step2-1;
	      for(stept2=step2+1;stept2!=step1+1;stept2++) {
		temp+=(*(sp1++))*(*sp2);
		if(stept2<hm1.num_row()) sp2+=stept2;
		}
	      if(step1<hm1.num_row()) {	// only if we aren't on the last row
		sp1+=step1-1;
		for(stept1=step1+1;stept1<=hm1.num_row();stept1++,stept2++) {
		  temp+=(*sp1)*(*sp2);
		  if(stept1<hm1.num_row()) {
	             sp1+=stept1;
		     sp2+=stept2;
		     }
		  }	// for(stept1
		}	// if(step1
	      }	// if(step2
	  }	// else
	*(mr++)=temp;
      }	// for(step2
    if(step1<hm1.num_row()) snp1+=step1;
    }	// for(step1
  return mret;
}

HepVector operator*(const HepSymMatrix &hm1,const HepVector &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepVector mret(hm1.num_row());
#endif
  CHK_DIM_1(hm1.num_col(),hm2.num_row(),*);
  HepMatrix::mcIter sp,snp,vpt;
  double temp;
  int step,stept;
  HepMatrix::mIter vrp=mret.m.begin();
  for(step=1,snp=hm1.m.begin();step<=hm1.num_row();++step)
    {
      sp=snp;
      vpt=hm2.m.begin();
      snp+=step;
      temp=0;
      while(sp<snp)
	temp+=*(sp++)*(*(vpt++));
      if(step<hm1.num_row()) sp+=step-1;
      for(stept=step+1;stept<=hm1.num_row();stept++)
	{ 
	  temp+=*sp*(*(vpt++));
	  if(stept<hm1.num_row()) sp+=stept;
	}
      *(vrp++)=temp;
    }	// for(step
  return mret;
}

HepSymMatrix vT_times_v(const HepVector &v)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(v.num_row());
{
#else
{
  HepSymMatrix mret(v.num_row());
#endif
  HepMatrix::mIter mr=mret.m.begin();
  HepMatrix::mcIter vt1,vt2;
  for(vt1=v.m.begin();vt1<v.m.begin()+v.num_row();vt1++)
    for(vt2=v.m.begin();vt2<=vt1;vt2++)
      *(mr++)=(*vt1)*(*vt2);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains the assignment and inplace operators =,+=,-=,*=,/=.
   ----------------------------------------------------------------------- */

HepMatrix & HepMatrix::operator+=(const HepSymMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  HepMatrix::mcIter sjk = hm2.m.begin();
  // j >= k
  for(int j=0; j!=nrow; ++j) {
     for(int k=0; k<=j; ++k) {
	m[j*ncol+k] += *sjk;
	// make sure this is not a diagonal element
	if(k!=j) m[k*nrow+j] += *sjk;
        ++sjk;
     } 
  }   
  return (*this);
}

HepSymMatrix & HepSymMatrix::operator+=(const HepSymMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

HepMatrix & HepMatrix::operator-=(const HepSymMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),-=);
  HepMatrix::mcIter sjk = hm2.m.begin();
  // j >= k
  for(int j=0; j!=nrow; ++j) {
     for(int k=0; k<=j; ++k) {
	m[j*ncol+k] -= *sjk;
	// make sure this is not a diagonal element
	if(k!=j) m[k*nrow+j] -= *sjk;
        ++sjk;
     } 
  }   
  return (*this);
}

HepSymMatrix & HepSymMatrix::operator-=(const HepSymMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

HepSymMatrix & HepSymMatrix::operator/=(double t)
{
  SIMPLE_UOP(/=)
  return (*this);
}

HepSymMatrix & HepSymMatrix::operator*=(double t)
{
  SIMPLE_UOP(*=)
  return (*this);
}

HepMatrix & HepMatrix::operator=(const HepSymMatrix &hm1)
{
   // define size, rows, and columns of *this
   nrow = ncol = hm1.nrow;
   if(nrow*ncol != size_)
   {
      size_ = nrow*ncol;
      m.resize(size_);
   }
   // begin copy
   mcIter sjk = hm1.m.begin();
   // j >= k
   for(int j=0; j!=nrow; ++j) {
      for(int k=0; k<=j; ++k) {
	 m[j*ncol+k] = *sjk;
	 // we could copy the diagonal element twice or check 
	 // doing the check may be a tiny bit faster,
	 // so we choose that option for now
	 if(k!=j) m[k*nrow+j] = *sjk;
         ++sjk;
      } 
   }   
   return (*this);
}

HepSymMatrix & HepSymMatrix::operator=(const HepSymMatrix &hm1)
{
   if(hm1.nrow != nrow)
   {
      nrow = hm1.nrow;
      size_ = hm1.size_;
      m.resize(size_);
   }
   m = hm1.m;
   return (*this);
}

HepSymMatrix & HepSymMatrix::operator=(const HepDiagMatrix &hm1)
{
   if(hm1.nrow != nrow)
   {
      nrow = hm1.nrow;
      size_ = nrow * (nrow+1) / 2;
      m.resize(size_);
   }

   m.assign(size_,0);
   HepMatrix::mIter mrr = m.begin();
   HepMatrix::mcIter mr = hm1.m.begin();
   for(int r=1; r<=nrow; r++) {
      *mrr = *(mr++);
      if(r<nrow) mrr += (r+1);
   }
   return (*this);
}

// Print the Matrix.

std::ostream& operator<<(std::ostream &os, const HepSymMatrix &q)
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
      for(int icol = 1; icol <= q.num_col(); icol++)
	{
	  os.width(width);
	  os << q(irow,icol) << " ";
	}
      os << std::endl;
    }
  return os;
}

HepSymMatrix HepSymMatrix::
apply(double (*f)(double, int, int)) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(num_row());
{
#else
{
  HepSymMatrix mret(num_row());
#endif
  HepMatrix::mcIter a = m.begin();
  HepMatrix::mIter b = mret.m.begin();
  for(int ir=1;ir<=num_row();ir++) {
    for(int ic=1;ic<=ir;ic++) {
      *(b++) = (*f)(*(a++), ir, ic);
    }
  }
  return mret;
}

void HepSymMatrix::assign (const HepMatrix &hm1)
{
   if(hm1.nrow != nrow)
   {
      nrow = hm1.nrow;
      size_ = nrow * (nrow+1) / 2;
      m.resize(size_);
   }
   HepMatrix::mcIter a = hm1.m.begin();
   HepMatrix::mIter b = m.begin();
   for(int r=1;r<=nrow;r++) {
      HepMatrix::mcIter d = a;
      for(int c=1;c<=r;c++) {
	 *(b++) = *(d++);
      }
      if(r<nrow) a += nrow;
   }
}

HepSymMatrix HepSymMatrix::similarity(const HepMatrix &hm1) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepSymMatrix mret(hm1.num_row());
#endif
  HepMatrix temp = hm1*(*this);
// If hm1*(*this) has correct dimensions, then so will the hm1.T multiplication.
// So there is no need to check dimensions again.
  int n = hm1.num_col();
  HepMatrix::mIter mr = mret.m.begin();
  HepMatrix::mIter tempr1 = temp.m.begin();
  for(int r=1;r<=mret.num_row();r++) {
    HepMatrix::mcIter hm1c1 = hm1.m.begin();
    for(int c=1;c<=r;c++) {
      double tmp = 0.0;
      HepMatrix::mIter tempri = tempr1;
      HepMatrix::mcIter hm1ci = hm1c1;
      for(int i=1;i<=hm1.num_col();i++) {
	tmp+=(*(tempri++))*(*(hm1ci++));
      }
      *(mr++) = tmp;
      hm1c1 += n;
    }
    tempr1 += n;
  }
  return mret;
}

HepSymMatrix HepSymMatrix::similarity(const HepSymMatrix &hm1) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row());
{
#else
{
  HepSymMatrix mret(hm1.num_row());
#endif
  HepMatrix temp = hm1*(*this);
  int n = hm1.num_col();
  HepMatrix::mIter mr = mret.m.begin();
  HepMatrix::mIter tempr1 = temp.m.begin();
  for(int r=1;r<=mret.num_row();r++) {
    HepMatrix::mcIter hm1c1 = hm1.m.begin();
    int c;
    for(c=1;c<=r;c++) {
      double tmp = 0.0;
      HepMatrix::mIter tempri = tempr1;
      HepMatrix::mcIter hm1ci = hm1c1;
      int i;
      for(i=1;i<c;i++) {
	tmp+=(*(tempri++))*(*(hm1ci++));
      }
      for(i=c;i<=hm1.num_col();i++) {
	tmp+=(*(tempri++))*(*(hm1ci));
	if(i<hm1.num_col()) hm1ci += i;
      }
      *(mr++) = tmp;
      hm1c1 += c;
    }
    tempr1 += n;
  }
  return mret;
}

double HepSymMatrix::similarity(const HepVector &hm1)
const {
  double mret = 0.0;
  HepVector temp = (*this) *hm1;
// If hm1*(*this) has correct dimensions, then so will the hm1.T multiplication.
// So there is no need to check dimensions again.
  HepMatrix::mIter a=temp.m.begin();
  HepMatrix::mcIter b=hm1.m.begin();
  HepMatrix::mIter e=a+hm1.num_row();
  for(;a<e;) mret += (*(a++)) * (*(b++));
  return mret;
}

HepSymMatrix HepSymMatrix::similarityT(const HepMatrix &hm1) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_col());
{
#else
{
  HepSymMatrix mret(hm1.num_col());
#endif
  HepMatrix temp = (*this)*hm1;
  int n = hm1.num_col();
  HepMatrix::mIter mrc = mret.m.begin();
  HepMatrix::mIter temp1r = temp.m.begin();
  for(int r=1;r<=mret.num_row();r++) {
    HepMatrix::mcIter m11c = hm1.m.begin();
    for(int c=1;c<=r;c++) {
      double tmp = 0.0;
      for(int i=1;i<=hm1.num_row();i++) {
	HepMatrix::mIter tempir = temp1r + n*(i-1);
	HepMatrix::mcIter hm1ic = m11c + n*(i-1);
	tmp+=(*(tempir))*(*(hm1ic));
      }
      *(mrc++) = tmp;
      m11c++;
    }
    temp1r++;
  }
  return mret;
}

void HepSymMatrix::invert(int &ifail) {
  
  ifail = 0;

  switch(nrow) {
  case 3:
    {
      double det, temp;
      double t1, t2, t3;
      double c11,c12,c13,c22,c23,c33;
      c11 = (*(m.begin()+2)) * (*(m.begin()+5)) - (*(m.begin()+4)) * (*(m.begin()+4));
      c12 = (*(m.begin()+4)) * (*(m.begin()+3)) - (*(m.begin()+1)) * (*(m.begin()+5));
      c13 = (*(m.begin()+1)) * (*(m.begin()+4)) - (*(m.begin()+2)) * (*(m.begin()+3));
      c22 = (*(m.begin()+5)) * (*m.begin()) - (*(m.begin()+3)) * (*(m.begin()+3));
      c23 = (*(m.begin()+3)) * (*(m.begin()+1)) - (*(m.begin()+4)) * (*m.begin());
      c33 = (*m.begin()) * (*(m.begin()+2)) - (*(m.begin()+1)) * (*(m.begin()+1));
      t1 = fabs(*m.begin());
      t2 = fabs(*(m.begin()+1));
      t3 = fabs(*(m.begin()+3));
      if (t1 >= t2) {
	if (t3 >= t1) {
	  temp = *(m.begin()+3);
	  det = c23*c12-c22*c13;
	} else {
	  temp = *m.begin();
	  det = c22*c33-c23*c23;
	}
      } else if (t3 >= t2) {
	temp = *(m.begin()+3);
	det = c23*c12-c22*c13;
      } else {
	temp = *(m.begin()+1);
	det = c13*c23-c12*c33;
      }
      if (det==0) {
	ifail = 1;
	return;
      }
      {
	double ds = temp/det;
	HepMatrix::mIter hmm = m.begin();
	*(hmm++) = ds*c11;
	*(hmm++) = ds*c12;
	*(hmm++) = ds*c22;
	*(hmm++) = ds*c13;
	*(hmm++) = ds*c23;
	*(hmm) = ds*c33;
      }
    }
    break;
 case 2:
    {
      double det, temp, ds;
      det = (*m.begin())*(*(m.begin()+2)) - (*(m.begin()+1))*(*(m.begin()+1));
      if (det==0) {
	ifail = 1;
	return;
      }
      ds = 1.0/det;
      *(m.begin()+1) *= -ds;
      temp = ds*(*(m.begin()+2));
      *(m.begin()+2) = ds*(*m.begin());
      *m.begin() = temp;
      break;
    }
 case 1:
    {
      if ((*m.begin())==0) {
	ifail = 1;
	return;
      }
      *m.begin() = 1.0/(*m.begin());
      break;
    }
 case 5:
    {
      invert5(ifail);
      return;
    }
 case 6:
    {
      invert6(ifail);
      return;
    }
 case 4:
    {
      invert4(ifail);
      return;
    }
 default:
    {
      invertBunchKaufman(ifail);
      return;
    }
  }
  return; // inversion successful
}

double HepSymMatrix::determinant() const {
  static const int max_array = 20;
  // ir must point to an array which is ***1 longer than*** nrow
  static std::vector<int> ir_vec (max_array+1); 
  if (ir_vec.size() <= static_cast<unsigned int>(nrow)) ir_vec.resize(nrow+1);
  int * ir = &ir_vec[0];   

  double det;
  HepMatrix mt(*this);
  int i = mt.dfact_matrix(det, ir);
  if(i==0) return det;
  return 0.0;
}

double HepSymMatrix::trace() const {
   double t = 0.0;
   for (int i=0; i<nrow; i++) 
     t += *(m.begin() + (i+3)*i/2);
   return t;
}

void HepSymMatrix::invertBunchKaufman(int &ifail) {
  // Bunch-Kaufman diagonal pivoting method
  // It is decribed in J.R. Bunch, L. Kaufman (1977). 
  // "Some Stable Methods for Calculating Inertia and Solving Symmetric 
  // Linear Systems", Math. Comp. 31, p. 162-179. or in Gene H. Golub, 
  // Charles F. van Loan, "Matrix Computations" (the second edition 
  // has a bug.) and implemented in "lapack"
  // Mario Stanke, 09/97

  int i, j, k, is;
  int pivrow;

  // Establish the two working-space arrays needed:  x and piv are
  // used as pointers to arrays of doubles and ints respectively, each
  // of length nrow.  We do not want to reallocate each time through
  // unless the size needs to grow.  We do not want to leak memory, even
  // by having a new without a delete that is only done once.
  
  static const int max_array = 25;
#ifdef DISABLE_ALLOC
  static std::vector<double> xvec (max_array);
  static std::vector<int>    pivv (max_array);
  typedef std::vector<int>::iterator pivIter; 
#else
  static std::vector<double,Alloc<double,25> > xvec (max_array);
  static std::vector<int,   Alloc<int,   25> > pivv (max_array);
  typedef std::vector<int,Alloc<int,25> >::iterator pivIter; 
#endif	
  if (xvec.size() < static_cast<unsigned int>(nrow)) xvec.resize(nrow);
  if (pivv.size() < static_cast<unsigned int>(nrow)) pivv.resize(nrow);
     // Note - resize shuld do  nothing if the size is already larger than nrow,
     //        but on VC++ there are indications that it does so we check.
     // Note - the data elements in a vector are guaranteed to be contiguous,
     //        so x[i] and piv[i] are optimally fast.
  mIter   x   = xvec.begin();
  // x[i] is used as helper storage, needs to have at least size nrow.
  pivIter piv = pivv.begin();
  // piv[i] is used to store details of exchanges
      
  double temp1, temp2;
  HepMatrix::mIter ip, mjj, iq;
  double lambda, sigma;
  const double alpha = .6404; // = (1+sqrt(17))/8
  const double epsilon = 32*DBL_EPSILON;
  // whenever a sum of two doubles is below or equal to epsilon
  // it is set to zero.
  // this constant could be set to zero but then the algorithm
  // doesn't neccessarily detect that a matrix is singular
  
  for (i = 0; i < nrow; ++i) piv[i] = i+1;
      
  ifail = 0;
      
  // compute the factorization P*A*P^T = L * D * L^T 
  // L is unit lower triangular, D is direct sum of 1x1 and 2x2 matrices
  // L and D^-1 are stored in A = *this, P is stored in piv[]
	
  for (j=1; j < nrow; j+=is)  // main loop over columns
  {
      mjj = m.begin() + j*(j-1)/2 + j-1;
      lambda = 0;           // compute lambda = max of A(j+1:n,j)
      pivrow = j+1;
      //ip = m.begin() + (j+1)*j/2 + j-1;
      for (i=j+1; i <= nrow ; ++i) {
          // calculate ip to avoid going off end of storage array
	  ip = m.begin() + (i-1)*i/2 + j-1;
	  if (fabs(*ip) > lambda) {
	      lambda = fabs(*ip);
	      pivrow = i;
	  }
      }	// for i
      if (lambda == 0 ) {
	  if (*mjj == 0) {
	      ifail = 1;
	      return;
	  }
	  is=1;
	  *mjj = 1./ *mjj;
      } else {	// lambda == 0
	  if (fabs(*mjj) >= lambda*alpha) {
	      is=1;
	      pivrow=j;
	  } else {	// fabs(*mjj) >= lambda*alpha
	      sigma = 0;  // compute sigma = max A(pivrow, j:pivrow-1)
	      ip = m.begin() + pivrow*(pivrow-1)/2+j-1;
	      for (k=j; k < pivrow; k++) {
		  if (fabs(*ip) > sigma) sigma = fabs(*ip);
		  ip++;
	      }	// for k
	      if (sigma * fabs(*mjj) >= alpha * lambda * lambda) {
		  is=1;
		  pivrow = j;
	      } else if (fabs(*(m.begin()+pivrow*(pivrow-1)/2+pivrow-1)) 
			    >= alpha * sigma) {
		is=1;
	      } else {
		is=2;
	      }	// if sigma...
	  }	// fabs(*mjj) >= lambda*alpha
	  if (pivrow == j) { // no permutation neccessary
	      piv[j-1] = pivrow;
	      if (*mjj == 0) {
		  ifail=1;
		  return;
	      }
	      temp2 = *mjj = 1./ *mjj; // invert D(j,j)

	      // update A(j+1:n, j+1,n)
	      for (i=j+1; i <= nrow; i++) {
		  temp1 = *(m.begin() + i*(i-1)/2 + j-1) * temp2;
		  ip = m.begin()+i*(i-1)/2+j;
		  for (k=j+1; k<=i; k++) {
		      *ip -= temp1 * *(m.begin() + k*(k-1)/2 + j-1);
		      if (fabs(*ip) <= epsilon)
			*ip=0;
		      ip++;
		  }
	      }	// for i
	      // update L 
	      //ip = m.begin() + (j+1)*j/2 + j-1; 
	      for (i=j+1; i <= nrow; ++i) {
        	  // calculate ip to avoid going off end of storage array
		  ip = m.begin() + (i-1)*i/2 + j-1;
	          *ip *= temp2;
	      }
	  } else if (is==1) { // 1x1 pivot 
	      piv[j-1] = pivrow;

	      // interchange rows and columns j and pivrow in
	      // submatrix (j:n,j:n)
	      ip = m.begin() + pivrow*(pivrow-1)/2 + j;
	      for (i=j+1; i < pivrow; i++, ip++) {
		  temp1 = *(m.begin() + i*(i-1)/2 + j-1);
		  *(m.begin() + i*(i-1)/2 + j-1)= *ip;
		  *ip = temp1;
	      }	// for i
	      temp1 = *mjj;
	      *mjj = *(m.begin()+pivrow*(pivrow-1)/2+pivrow-1);
	      *(m.begin()+pivrow*(pivrow-1)/2+pivrow-1) = temp1;
	      ip = m.begin() + (pivrow+1)*pivrow/2 + j-1;
	      iq = ip + pivrow-j;
	      for (i = pivrow+1; i <= nrow; ip += i, iq += i++) {
		  temp1 = *iq;
		  *iq = *ip;
		  *ip = temp1;
	      }	// for i

	      if (*mjj == 0) {
		  ifail = 1;
		  return;
	      }	// *mjj == 0
	      temp2 = *mjj = 1./ *mjj; // invert D(j,j)

	      // update A(j+1:n, j+1:n)
	      for (i = j+1; i <= nrow; i++) {
		  temp1 = *(m.begin() + i*(i-1)/2 + j-1) * temp2;
		  ip = m.begin()+i*(i-1)/2+j;
		  for (k=j+1; k<=i; k++) {
		      *ip -= temp1 * *(m.begin() + k*(k-1)/2 + j-1);
		      if (fabs(*ip) <= epsilon)
			*ip=0;
		      ip++;
		  }	// for k
	      }	// for i
	      // update L 
	      //ip = m.begin() + (j+1)*j/2 + j-1; 
	      for (i=j+1; i <= nrow; ++i) {
        	  // calculate ip to avoid going off end of storage array
		  ip = m.begin() + (i-1)*i/2 + j-1;
	          *ip *= temp2;
	      }
	  } else { // is=2, ie use a 2x2 pivot
	      piv[j-1] = -pivrow;
	      piv[j] = 0; // that means this is the second row of a 2x2 pivot

	      if (j+1 != pivrow) {
		  // interchange rows and columns j+1 and pivrow in
		  // submatrix (j:n,j:n) 
		  ip = m.begin() + pivrow*(pivrow-1)/2 + j+1;
		  for (i=j+2; i < pivrow; i++, ip++) {
		      temp1 = *(m.begin() + i*(i-1)/2 + j);
		      *(m.begin() + i*(i-1)/2 + j) = *ip;
		      *ip = temp1;
		  }	// for i
		  temp1 = *(mjj + j + 1);
		  *(mjj + j + 1) = 
		    *(m.begin() + pivrow*(pivrow-1)/2 + pivrow-1);
		  *(m.begin() + pivrow*(pivrow-1)/2 + pivrow-1) = temp1;
		  temp1 = *(mjj + j);
		  *(mjj + j) = *(m.begin() + pivrow*(pivrow-1)/2 + j-1);
		  *(m.begin() + pivrow*(pivrow-1)/2 + j-1) = temp1;
		  ip = m.begin() + (pivrow+1)*pivrow/2 + j;
		  iq = ip + pivrow-(j+1);
		  for (i = pivrow+1; i <= nrow; ip += i, iq += i++) {
		      temp1 = *iq;
		      *iq = *ip;
		      *ip = temp1;
		  }	// for i
	      }	//  j+1 != pivrow
	      // invert D(j:j+1,j:j+1)
	      temp2 = *mjj * *(mjj + j + 1) - *(mjj + j) * *(mjj + j); 
	      if (temp2 == 0) {
		std::cerr
		  << "SymMatrix::bunch_invert: error in pivot choice" 
		  << std::endl;
              }
	      temp2 = 1. / temp2;
	      // this quotient is guaranteed to exist by the choice 
	      // of the pivot
	      temp1 = *mjj;
	      *mjj = *(mjj + j + 1) * temp2;
	      *(mjj + j + 1) = temp1 * temp2;
	      *(mjj + j) = - *(mjj + j) * temp2;

	      if (j < nrow-1) { // otherwise do nothing
		  // update A(j+2:n, j+2:n)
		  for (i=j+2; i <= nrow ; i++) {
		      ip = m.begin() + i*(i-1)/2 + j-1;
		      temp1 = *ip * *mjj + *(ip + 1) * *(mjj + j);
		      if (fabs(temp1 ) <= epsilon) temp1 = 0;
		      temp2 = *ip * *(mjj + j) + *(ip + 1) * *(mjj + j + 1);
		      if (fabs(temp2 ) <= epsilon) temp2 = 0;
		      for (k = j+2; k <= i ; k++) {
			  ip = m.begin() + i*(i-1)/2 + k-1;
			  iq = m.begin() + k*(k-1)/2 + j-1;
			  *ip -= temp1 * *iq + temp2 * *(iq+1);
			  if (fabs(*ip) <= epsilon)
			    *ip = 0;
		      }	// for k
		  }	// for i
		  // update L
		  for (i=j+2; i <= nrow ; i++) {
		      ip = m.begin() + i*(i-1)/2 + j-1;
		      temp1 = *ip * *mjj + *(ip+1) * *(mjj + j);
		      if (fabs(temp1) <= epsilon) temp1 = 0;
		      *(ip+1) = *ip * *(mjj + j) 
			        + *(ip+1) * *(mjj + j + 1);
		      if (fabs(*(ip+1)) <= epsilon) *(ip+1) = 0;
		      *ip = temp1;
		  }	// for k
	      }	// j < nrow-1
	  }
      }
  } // end of main loop over columns

  if (j == nrow) { // the the last pivot is 1x1
      mjj = m.begin() + j*(j-1)/2 + j-1;
      if (*mjj == 0) {
	  ifail = 1;
	  return;
      } else { *mjj = 1. / *mjj; }
  } // end of last pivot code

  // computing the inverse from the factorization
	 
  for (j = nrow ; j >= 1 ; j -= is) // loop over columns
  {
      mjj = m.begin() + j*(j-1)/2 + j-1;
      if (piv[j-1] > 0) { // 1x1 pivot, compute column j of inverse
	  is = 1; 
	  if (j < nrow) {
	      //ip = m.begin() + (j+1)*j/2 + j-1;
	      //for (i=0; i < nrow-j; ip += 1+j+i++) x[i] = *ip;
	      ip = m.begin() + (j+1)*j/2 - 1;
	      for (i=0; i < nrow-j; ++i) {
	          ip += i + j;
	          x[i] = *ip;
	      }
	      for (i=j+1; i<=nrow ; i++) {
		  temp2=0;
		  ip = m.begin() + i*(i-1)/2 + j;
		  for (k=0; k <= i-j-1; k++) temp2 += *ip++ * x[k];
		  // avoid setting ip outside the bounds of the storage array
		  ip -= 1;
		  // using the value of k from the previous loop
		  for ( ; k < nrow-j; ++k) {
		      ip += j+k;
		      temp2 += *ip * x[k];
		  }
		  *(m.begin()+ i*(i-1)/2 + j-1) = -temp2;
	      }	// for i
	      temp2 = 0;
	      //ip = m.begin() + (j+1)*j/2 + j-1;
	      //for (k=0; k < nrow-j; ip += 1+j+k++)
		//temp2 += x[k] * *ip;
	      ip = m.begin() + (j+1)*j/2 - 1;
	      for (k=0; k < nrow-j; ++k) {
	        ip += j+k;
		temp2 += x[k] * *ip;
	      }
	      *mjj -= temp2;
	  }	// j < nrow
      } else { //2x2 pivot, compute columns j and j-1 of the inverse
	  if (piv[j-1] != 0)
	    std::cerr << "error in piv" << piv[j-1] << std::endl;
	  is=2; 
	  if (j < nrow) {
	      //ip = m.begin() + (j+1)*j/2 + j-1;
	      //for (i=0; i < nrow-j; ip += 1+j+i++) x[i] = *ip;
	      ip = m.begin() + (j+1)*j/2 - 1;
	      for (i=0; i < nrow-j; ++i) {
	          ip += i + j;
	          x[i] = *ip;
	      }
	      for (i=j+1; i<=nrow ; i++) {
		  temp2 = 0;
		  ip = m.begin() + i*(i-1)/2 + j;
		  for (k=0; k <= i-j-1; k++)
		    temp2 += *ip++ * x[k];
		  for (ip += i-1; k < nrow-j; ip += 1+j+k++)
		    temp2 += *ip * x[k];
		  *(m.begin()+ i*(i-1)/2 + j-1) = -temp2;
	      }	// for i   
	      temp2 = 0;
	      //ip = m.begin() + (j+1)*j/2 + j-1;
	      //for (k=0; k < nrow-j; ip += 1+j+k++) temp2 += x[k] * *ip;
	      ip = m.begin() + (j+1)*j/2 - 1;
	      for (k=0; k < nrow-j; ++k) {
	          ip += k + j;
	          temp2 += x[k] * *ip;
	      }
	      *mjj -= temp2;
	      temp2 = 0;
	      //ip = m.begin() + (j+1)*j/2 + j-2;
	      //for (i=j+1; i <= nrow; ip += i++) temp2 += *ip * *(ip+1);
	      ip = m.begin() + (j+1)*j/2 - 2;
	      for (i=j+1; i <= nrow; ++i) {
	          ip += i - 1;
	          temp2 += *ip * *(ip+1);
	      }
	      *(mjj-1) -= temp2;
	      //ip = m.begin() + (j+1)*j/2 + j-2;
	      //for (i=0; i < nrow-j; ip += 1+j+i++) x[i] = *ip;
	      ip = m.begin() + (j+1)*j/2 - 2;
	      for (i=0; i < nrow-j; ++i) {
	          ip += i + j;
	          x[i] = *ip;
	      }
	      for (i=j+1; i <= nrow ; i++) {
		  temp2 = 0;
		  ip = m.begin() + i*(i-1)/2 + j;
		  for (k=0; k <= i-j-1; k++)
		      temp2 += *ip++ * x[k];
		  for (ip += i-1; k < nrow-j; ip += 1+j+k++)
		      temp2 += *ip * x[k];
		  *(m.begin()+ i*(i-1)/2 + j-2)= -temp2;
	      }	// for i
	      temp2 = 0;
	      //ip = m.begin() + (j+1)*j/2 + j-2;
	      //for (k=0; k < nrow-j; ip += 1+j+k++)
		//  temp2 += x[k] * *ip;
	      ip = m.begin() + (j+1)*j/2 - 2;
	      for (k=0; k < nrow-j; ++k) {
	          ip += k + j;
		  temp2 += x[k] * *ip;
	      }
	      *(mjj-j) -= temp2;
	  }	// j < nrow
      }	// else  piv[j-1] > 0

      // interchange rows and columns j and piv[j-1] 
      // or rows and columns j and -piv[j-2]

      pivrow = (piv[j-1]==0)? -piv[j-2] : piv[j-1];
      ip = m.begin() + pivrow*(pivrow-1)/2 + j;
      for (i=j+1;i < pivrow; i++, ip++) {
	  temp1 = *(m.begin() + i*(i-1)/2 + j-1);
	  *(m.begin() + i*(i-1)/2 + j-1) = *ip;
	  *ip = temp1;
      }	// for i
      temp1 = *mjj;
      *mjj = *(m.begin() + pivrow*(pivrow-1)/2 + pivrow-1);
      *(m.begin() + pivrow*(pivrow-1)/2 + pivrow-1) = temp1;
      if (is==2) {
	  temp1 = *(mjj-1);
	  *(mjj-1) = *( m.begin() + pivrow*(pivrow-1)/2 + j-2);
	  *( m.begin() + pivrow*(pivrow-1)/2 + j-2) = temp1;
      }	// is==2

      // problem right here
      if( pivrow < nrow ) {
	  ip = m.begin() + (pivrow+1)*pivrow/2 + j-1;  // &A(i,j)
	  // adding parenthesis for VC++
	  iq = ip + (pivrow-j);
	  for (i = pivrow+1; i <= nrow; i++) {
	      temp1 = *iq;
	      *iq = *ip;
	      *ip = temp1;
	      if( i < nrow ) {
	      ip += i;
	      iq += i;
	      }
	  }	// for i 
      }	// pivrow < nrow
  } // end of loop over columns (in computing inverse from factorization)

  return; // inversion successful

}

}  // namespace CLHEP
