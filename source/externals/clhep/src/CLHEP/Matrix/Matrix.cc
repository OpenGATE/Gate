// -*- C++ -*-
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//

#include <iostream>
#include <string.h>
#include <float.h>        // for DBL_EPSILON
#include <cmath>
#include <stdlib.h>

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/DiagMatrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Utility/thread_local.h"

#ifdef HEP_DEBUG_INLINE
#include "CLHEP/Matrix/Matrix.icc"
#endif

namespace CLHEP {

// Simple operation for all elements

#define SIMPLE_UOP(OPER)                            \
   mIter a=m.begin();                      \
   mIter e=m.end();                        \
   for(;a!=e; a++) (*a) OPER t;

#define SIMPLE_BOP(OPER)                            \
   HepMatrix::mIter a=m.begin();                      \
   HepMatrix::mcIter b=hm2.m.begin();                  \
   HepMatrix::mIter e=m.end();                        \
   for(;a!=e; a++, b++) (*a) OPER (*b);

#define SIMPLE_TOP(OPER)                            \
   HepMatrix::mcIter a=hm1.m.begin();       \
   HepMatrix::mcIter b=hm2.m.begin();       \
   HepMatrix::mIter t=mret.m.begin();      \
   HepMatrix::mcIter e=hm1.m.end();         \
   for(;a!=e; a++, b++, t++) (*t) = (*a) OPER (*b);

// Static functions.

#define CHK_DIM_2(r1,r2,c1,c2,fun) \
   if (r1!=r2 || c1!=c2)  { \
     HepGenMatrix::error("Range error in Matrix function " #fun "(1)."); \
   }

#define CHK_DIM_1(c1,r2,fun) \
   if (c1!=r2) { \
     HepGenMatrix::error("Range error in Matrix function " #fun "(2)."); \
   }

// Constructors. (Default constructors are inlined and in .icc file)

HepMatrix::HepMatrix(int p,int q)
   : m(p*q), nrow(p), ncol(q)
{
  size_ = nrow * ncol;
}

HepMatrix::HepMatrix(int p,int q,int init)
   : m(p*q), nrow(p), ncol(q)
{
   size_ = nrow * ncol;

   if (size_ > 0) {
      switch(init)
      {
      case 0:
	 break;

      case 1:
	 {
	    if ( ncol == nrow ) {
 	       mIter a = m.begin();
	       for( int step=0; step < size_; step+=(ncol+1) ) *(a+step) = 1.0;
	    } else {
	       error("Invalid dimension in HepMatrix(int,int,1).");
	    }
	    break;
	 }
      default:
	 error("Matrix: initialization must be either 0 or 1.");
      }
   }
}

HepMatrix::HepMatrix(int p,int q, HepRandom &r)
   : m(p*q), nrow(p), ncol(q)
{
   size_ = nrow * ncol;

   mIter a = m.begin();
   mIter b = m.end();
   for(; a<b; a++) *a = r();
}
//
// Destructor
//
HepMatrix::~HepMatrix() {
}

HepMatrix::HepMatrix(const HepMatrix &hm1)
   : HepGenMatrix(hm1), m(hm1.size_), nrow(hm1.nrow), ncol(hm1.ncol), size_(hm1.size_)
{
   m = hm1.m;

}

// trivial

int HepMatrix::num_row() const { return nrow;}

int HepMatrix::num_col() const  { return ncol;}

int HepMatrix::num_size() const { return size_;}

// operator()

double & HepMatrix::operator()(int row, int col)
{
#ifdef MATRIX_BOUND_CHECK
  if(row<1 || row>num_row() || col<1 || col>num_col())
    error("Range error in HepMatrix::operator()");
#endif
  return *(m.begin()+(row-1)*ncol+col-1);
}

const double & HepMatrix::operator()(int row, int col) const 
{
#ifdef MATRIX_BOUND_CHECK
  if(row<1 || row>num_row() || col<1 || col>num_col())
    error("Range error in HepMatrix::operator()");
#endif
  return *(m.begin()+(row-1)*ncol+col-1);
}


HepMatrix::HepMatrix(const HepSymMatrix &hm1)
   : m(hm1.nrow*hm1.nrow), nrow(hm1.nrow), ncol(hm1.nrow)
{
   size_ = nrow * ncol;

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
}

HepMatrix::HepMatrix(const HepDiagMatrix &hm1)
   : m(hm1.nrow*hm1.nrow), nrow(hm1.nrow), ncol(hm1.nrow)
{
   size_ = nrow * ncol;

   int n = num_row();
   mIter mrr;
   mcIter mr = hm1.m.begin();
   for(int r=0;r<n;r++) {
      mrr = m.begin()+(n+1)*r;
      *mrr = *(mr++);
   }
}

HepMatrix::HepMatrix(const HepVector &hm1)
   : m(hm1.nrow), nrow(hm1.nrow), ncol(1)
{

   size_ = nrow;
   m = hm1.m;
}


//
//
// Sub matrix
//
//

HepMatrix HepMatrix::sub(int min_row, int max_row,
			 int min_col,int max_col) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(max_row-min_row+1,max_col-min_col+1);
{
#else
{
  HepMatrix mret(max_row-min_row+1,max_col-min_col+1);
#endif
  if(max_row > num_row() || max_col >num_col())
    error("HepMatrix::sub: Index out of range");
  mIter a = mret.m.begin();
  int nc = num_col();
  mcIter b1 = m.begin() + (min_row - 1) * nc + min_col - 1;
  int rowsize = mret.num_row();
  for(int irow=1; irow<=rowsize; ++irow) {
    mcIter brc = b1;
    for(int icol=0; icol<mret.num_col(); ++icol) {
      *(a++) = *(brc++);
    }
    if(irow<rowsize) b1 += nc;
  }
  return mret;
}

void HepMatrix::sub(int row,int col,const HepMatrix &hm1)
{
  if(row <1 || row+hm1.num_row()-1 > num_row() || 
     col <1 || col+hm1.num_col()-1 > num_col()   )
    error("HepMatrix::sub: Index out of range");
  mcIter a = hm1.m.begin();
  int nc = num_col();
  mIter b1 = m.begin() + (row - 1) * nc + col - 1;
  int rowsize = hm1.num_row();
  for(int irow=1; irow<=rowsize; ++irow) {
    mIter brc = b1;
    for(int icol=0; icol<hm1.num_col(); ++icol) {
      *(brc++) = *(a++);
    }
    if(irow<rowsize) b1 += nc;
  }
}

//
// Direct sum of two matricies
//

HepMatrix dsum(const HepMatrix &hm1, const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
  return mret(hm1.num_row() + hm2.num_row(), hm1.num_col() + hm2.num_col(),
	      0);
{
#else
{
  HepMatrix mret(hm1.num_row() + hm2.num_row(), hm1.num_col() + hm2.num_col(),
		 0);
#endif
  mret.sub(1,1,hm1);
  mret.sub(hm1.num_row()+1,hm1.num_col()+1,hm2);
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This section contains
   The two argument functions +,-. They call the copy constructor and +=,-=.
   ----------------------------------------------------------------------- */
HepMatrix HepMatrix::operator- () const 
#ifdef HEP_GNU_OPTIMIZED_RETURN
      return hm2(nrow, ncol);
{
#else
{
   HepMatrix hm2(nrow, ncol);
#endif
   mcIter a=m.begin();
   mIter b=hm2.m.begin();
   mcIter e=m.end();
   for(;a<e; a++, b++) (*b) = -(*a);
   return hm2;
}

   

HepMatrix operator+(const HepMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.nrow, hm1.ncol);
{
#else
{
  HepMatrix mret(hm1.nrow, hm1.ncol);
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(), hm1.num_col(),hm2.num_col(),+);
  SIMPLE_TOP(+)
  return mret;
}

//
// operator -
//

HepMatrix operator-(const HepMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.num_row(), hm1.num_col());
{
#else
{
  HepMatrix mret(hm1.num_row(), hm1.num_col());
#endif
  CHK_DIM_2(hm1.num_row(),hm2.num_row(),
			 hm1.num_col(),hm2.num_col(),-);
  SIMPLE_TOP(-)
  return mret;
}

/* -----------------------------------------------------------------------
   This section contains support routines for matrix.h. This file contains
   The two argument functions *,/. They call copy constructor and then /=,*=.
   ----------------------------------------------------------------------- */

HepMatrix operator/(
const HepMatrix &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  mret /= t;
  return mret;
}

HepMatrix operator*(const HepMatrix &hm1,double t)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepMatrix operator*(double t,const HepMatrix &hm1)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1);
{
#else
{
  HepMatrix mret(hm1);
#endif
  mret *= t;
  return mret;
}

HepMatrix operator*(const HepMatrix &hm1,const HepMatrix &hm2)
#ifdef HEP_GNU_OPTIMIZED_RETURN
     return mret(hm1.nrow,hm2.ncol,0);
{
#else
{
  // initialize matrix to 0.0
  HepMatrix mret(hm1.nrow,hm2.ncol,0);
#endif
  CHK_DIM_1(hm1.ncol,hm2.nrow,*);

  int m1cols = hm1.ncol;
  int m2cols = hm2.ncol;

  for (int i=0; i<hm1.nrow; i++)
  {
     for (int j=0; j<m1cols; j++) 
     {
	double temp = hm1.m[i*m1cols+j];
	HepMatrix::mIter pt = mret.m.begin() + i*m2cols;
	
	// Loop over k (the column index in matrix hm2)
	HepMatrix::mcIter pb = hm2.m.begin() + m2cols*j;
	const HepMatrix::mcIter pblast = pb + m2cols;
	while (pb < pblast)
	{
	   (*pt) += temp * (*pb);
	   pb++;
	   pt++;
	}
     }
  }

  return mret;
}

/* -----------------------------------------------------------------------
   This section contains the assignment and inplace operators =,+=,-=,*=,/=.
   ----------------------------------------------------------------------- */

HepMatrix & HepMatrix::operator+=(const HepMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),+=);
  SIMPLE_BOP(+=)
  return (*this);
}

HepMatrix & HepMatrix::operator-=(const HepMatrix &hm2)
{
  CHK_DIM_2(num_row(),hm2.num_row(),num_col(),hm2.num_col(),-=);
  SIMPLE_BOP(-=)
  return (*this);
}

HepMatrix & HepMatrix::operator/=(double t)
{
  SIMPLE_UOP(/=)
  return (*this);
}

HepMatrix & HepMatrix::operator*=(double t)
{
  SIMPLE_UOP(*=)
  return (*this);
}

HepMatrix & HepMatrix::operator=(const HepMatrix &hm1)
{
   if(hm1.nrow*hm1.ncol != size_) //??fixme?? hm1.size != size
   {
      size_ = hm1.nrow * hm1.ncol;
      m.resize(size_); //??fixme?? if (size < hm1.size) m.resize(hm1.size);
   }
   nrow = hm1.nrow;
   ncol = hm1.ncol;
   m = hm1.m;
   return (*this);
}

// HepMatrix & HepMatrix::operator=(const HepRotation &hm2) 
// is now in Matrix=Rotation.cc

// Print the Matrix.

std::ostream& operator<<(std::ostream &os, const HepMatrix &q)
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

HepMatrix HepMatrix::T() const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(ncol,nrow);
{
#else
{
   HepMatrix mret(ncol,nrow);
#endif
   mcIter pme = m.begin();
   mIter pt = mret.m.begin();
   for( int nr=0; nr<nrow; ++nr) {
       for( int nc=0; nc<ncol; ++nc) {
          pt = mret.m.begin() + nr + nrow*nc;
          (*pt) = (*pme);
          ++pme;
       }
   }
   return mret;
}

HepMatrix HepMatrix::apply(double (*f)(double, int, int)) const
#ifdef HEP_GNU_OPTIMIZED_RETURN
return mret(num_row(),num_col());
{
#else
{
  HepMatrix mret(num_row(),num_col());
#endif
  mcIter a = m.begin();
  mIter b = mret.m.begin();
  for(int ir=1;ir<=num_row();ir++) {
    for(int ic=1;ic<=num_col();ic++) {
      *(b++) = (*f)(*(a++), ir, ic);
    }
  }
  return mret;
}

int HepMatrix::dfinv_matrix(int *ir) {
  if (num_col()!=num_row())
    error("dfinv_matrix: Matrix is not NxN");
  int n = num_col();
  if (n==1) return 0;

  double s31, s32;
  double s33, s34;

  mIter hm11 = m.begin();
  mIter hm12 = hm11 + 1;
  mIter hm21 = hm11 + n;
  mIter hm22 = hm12 + n;
  *hm21 = -(*hm22) * (*hm11) * (*hm21);
  *hm12 = -(*hm12);
  if (n>2) {
    mIter mimim = hm11 + n + 1;
    for (int i=3;i<=n;i++) {
      // calculate these to avoid pointing off the end of the storage array
      mIter mi = hm11 + (i-1) * n;
      mIter mii= hm11 + (i-1) * n + i - 1;
      int ihm2 = i - 2;
      mIter mj = hm11;
      mIter mji = mj + i - 1;
      mIter mij = mi;
      for (int j=1;j<=ihm2;j++) { 
	s31 = 0.0;
	s32 = *mji;
	mIter mkj = mj + j - 1;
	mIter mik = mi + j - 1;
	mIter mjkp = mj + j;
	mIter mkpi = mj + n + i - 1;
	for (int k=j;k<=ihm2;k++) {
	  s31 += (*mkj) * (*(mik++));
	  s32 += (*(mjkp++)) * (*mkpi);
	  mkj += n;
	  mkpi += n;
	}	// for k
	*mij = -(*mii) * (((*(mij-n)))*( (*(mii-1)))+(s31));
	*mji = -s32;
	mj += n;
	mji += n;
	mij++;
      }	// for j
      *(mii-1) = -(*mii) * (*mimim) * (*(mii-1));
      *(mimim+1) = -(*(mimim+1));
      mimim += (n+1);
    }	// for i
  }	// n>2
  mIter mi = hm11;
  mIter mii = hm11;
  for (int i=1;i<n;i++) {
    int ni = n - i;
    mIter mij = mi;
    int j;
    for (j=1; j<=i;j++) {
      s33 = *mij;
      // change initial definition of mikj to avoid pointing off the end of the storage array
      mIter mikj = mi + j - 1;
      mIter miik = mii + 1;
      mIter min_end = mi + n;
      for (;miik<min_end;) {
        // iterate by n as we enter the loop to avoid pointing off the end of the storage array
	mikj += n;
	s33 += (*mikj) * (*(miik++));
      }
      *(mij++) = s33;
    }
    for (j=1;j<=ni;j++) {
      s34 = 0.0;
      mIter miik = mii + j;
      for (int k=j;k<=ni;k++) {
        // calculate mikij here to avoid pointing off the end of the storage array
        mIter mikij = mii + k * n + j;
	s34 += *mikij * (*(miik++));
      }
      *(mii+j) = s34;
    }
    mi += n;
    mii += (n+1);
  }	// for i
  int nxch = ir[n];
  if (nxch==0) return 0;
  for (int hmm=1;hmm<=nxch;hmm++) {
    int k = nxch - hmm + 1;
    int ij = ir[k];
    int i = ij >> 12;
    int j = ij%4096;
    for (k=1; k<=n;k++) {
      // avoid setting the iterator beyond the end of the storage vector
      mIter mki = hm11 + (k-1)*n + i - 1;
      mIter mkj = hm11 + (k-1)*n + j - 1;
      // 2/24/05 David Sachs fix of improper swap bug that was present
      // for many years:
      double ti = *mki; // 2/24/05
      *mki = *mkj;
      *mkj = ti;	// 2/24/05
    }
  }	// for hmm
  return 0;
}

int HepMatrix::dfact_matrix(double &det, int *ir) {
  if (ncol!=nrow)
     error("dfact_matrix: Matrix is not NxN");

  int ifail, jfail;
  int n = ncol;

  double tf;
  double g1 = 1.0e-19, g2 = 1.0e19;

  double p, q, t;
  double s11, s12;

  double epsilon = 8*DBL_EPSILON;
  // could be set to zero (like it was before)
  // but then the algorithm often doesn't detect
  // that a matrix is singular

  int normal = 0, imposs = -1;
  int jrange = 0, jover = 1, junder = -1;
  ifail = normal;
  jfail = jrange;
  int nxch = 0;
  det = 1.0;
  mIter mj = m.begin();
  mIter mjj = mj;
  for (int j=1;j<=n;j++) {
    int k = j;
    p = (fabs(*mjj));
    if (j!=n) {
      // replace mij with calculation of position
      for (int i=j+1;i<n;i++) {
	q = (fabs(*(mj + n*(i-j) + j - 1)));
	if (q > p) {
	  k = i;
	  p = q;
	}
      }	// for i
      if (k==j) {
	if (p <= epsilon) {
	  det = 0;
	  ifail = imposs;
	  jfail = jrange;
	  return ifail;
	}
	det = -det; // in this case the sign of the determinant
	            // must not change. So I change it twice. 
      }	// k==j
      mIter mjl = mj;
      mIter mkl = m.begin() + (k-1)*n;
      for (int l=1;l<=n;l++) {
        tf = *mjl;
        *(mjl++) = *mkl;
        *(mkl++) = tf;
      }
      nxch = nxch + 1;  // this makes the determinant change its sign
      ir[nxch] = (((j)<<12)+(k));
    } else {	// j!=n
      if (p <= epsilon) {
	det = 0.0;
	ifail = imposs;
	jfail = jrange;
	return ifail;
      }
    }	// j!=n
    det *= *mjj;
    *mjj = 1.0 / *mjj;
    t = (fabs(det));
    if (t < g1) {
      det = 0.0;
      if (jfail == jrange) jfail = junder;
    } else if (t > g2) {
      det = 1.0;
      if (jfail==jrange) jfail = jover;
    }
    // calculate mk and mkjp so we don't point off the end of the vector
    if (j!=n) {
      mIter mjk = mj + j;
      for (k=j+1;k<=n;k++) {
	mIter mk = mj + n*(k-j);
	mIter mkjp = mk + j;
	s11 = - (*mjk);
	s12 = - (*mkjp);
	if (j!=1) {
	  mIter mik = m.begin() + k - 1;
	  mIter mijp = m.begin() + j;
	  mIter mki = mk;
	  mIter mji = mj;
	  for (int i=1;i<j;i++) {
	    s11 += (*mik) * (*(mji++));
	    s12 += (*mijp) * (*(mki++));
	    mik += n;
	    mijp += n;
	  }  // for i
	} // j!=1
	*(mjk++) = -s11 * (*mjj);
	*(mkjp) = -(((*(mjj+1)))*((*(mkjp-1)))+(s12));
      } // for k
    } // j!=n
    // avoid setting the iterator beyond the end of the vector
    if(j!=n) {
      mj += n;
      mjj += (n+1);
    }
  }	// for j
  if (nxch%2==1) det = -det;
  if (jfail !=jrange) det = 0.0;
  ir[n] = nxch;
  return 0;
}

void HepMatrix::invert(int &ierr) {
  if(ncol != nrow)
     error("HepMatrix::invert: Matrix is not NxN");

  static CLHEP_THREAD_LOCAL int max_array = 20;
  static CLHEP_THREAD_LOCAL int *ir = new int [max_array+1];

  if (ncol > max_array) {
    delete [] ir;
    max_array = nrow;
    ir = new int [max_array+1];
  }
  double t1, t2, t3;
  double det, temp, sd;
  int ifail;
  switch(nrow) {
  case 3:
    double c11,c12,c13,c21,c22,c23,c31,c32,c33;
    ifail = 0;
    c11 = (*(m.begin()+4)) * (*(m.begin()+8)) - (*(m.begin()+5)) * (*(m.begin()+7));
    c12 = (*(m.begin()+5)) * (*(m.begin()+6)) - (*(m.begin()+3)) * (*(m.begin()+8));
    c13 = (*(m.begin()+3)) * (*(m.begin()+7)) - (*(m.begin()+4)) * (*(m.begin()+6));
    c21 = (*(m.begin()+7)) * (*(m.begin()+2)) - (*(m.begin()+8)) * (*(m.begin()+1));
    c22 = (*(m.begin()+8)) * (*m.begin()) - (*(m.begin()+6)) * (*(m.begin()+2));
    c23 = (*(m.begin()+6)) * (*(m.begin()+1)) - (*(m.begin()+7)) * (*m.begin());
    c31 = (*(m.begin()+1)) * (*(m.begin()+5)) - (*(m.begin()+2)) * (*(m.begin()+4));
    c32 = (*(m.begin()+2)) * (*(m.begin()+3)) - (*m.begin()) * (*(m.begin()+5));
    c33 = (*m.begin()) * (*(m.begin()+4)) - (*(m.begin()+1)) * (*(m.begin()+3));
    t1 = fabs(*m.begin());
    t2 = fabs(*(m.begin()+3));
    t3 = fabs(*(m.begin()+6));
    if (t1 >= t2) {
      if (t3 >= t1) {
      temp = *(m.begin()+6);
      det = c23*c12-c22*c13;
      } else {
	temp = *(m.begin());
	det = c22*c33-c23*c32;
      }
    } else if (t3 >= t2) {
      temp = *(m.begin()+6);
      det = c23*c12-c22*c13;
    } else {
      temp = *(m.begin()+3);
      det = c13*c32-c12*c33;
    }
    if (det==0) {
      ierr = 1;
      return;
    }
    {
      double s1 = temp/det;
      mIter hmm = m.begin();
      *(hmm++) = s1*c11;
      *(hmm++) = s1*c21;
      *(hmm++) = s1*c31;
      *(hmm++) = s1*c12;
      *(hmm++) = s1*c22;
      *(hmm++) = s1*c32;
      *(hmm++) = s1*c13;
      *(hmm++) = s1*c23;
      *(hmm) = s1*c33;
    }
    break;
  case 2:
    ifail = 0;
    det = (*m.begin())*(*(m.begin()+3)) - (*(m.begin()+1))*(*(m.begin()+2));
    if (det==0) {
      ierr = 1;
      return;
    }
    sd = 1.0/det;
    temp = sd*(*(m.begin()+3));
    *(m.begin()+1) *= -sd;
    *(m.begin()+2) *= -sd;
    *(m.begin()+3) = sd*(*m.begin());
    *(m.begin()) = temp;
    break;
  case 1:
    ifail = 0;
    if ((*(m.begin()))==0) {
      ierr = 1;
      return;
    }
    *(m.begin()) = 1.0/(*(m.begin()));
    break;
  case 4:
    invertHaywood4(ierr);
    return;
  case 5:
    invertHaywood5(ierr);
    return;
  case 6:
    invertHaywood6(ierr);
    return;
  default:
    ifail = dfact_matrix(det, ir);
    if(ifail) {
      ierr = 1;
      return;
    }
    dfinv_matrix(ir);
    break;
  }
  ierr = 0;
  return;
}

double HepMatrix::determinant() const {
  static CLHEP_THREAD_LOCAL int max_array = 20;
  static CLHEP_THREAD_LOCAL int *ir = new int [max_array+1];
  if(ncol != nrow)
    error("HepMatrix::determinant: Matrix is not NxN");
  if (ncol > max_array) {
    delete [] ir;
    max_array = nrow;
    ir = new int [max_array+1];
  }
  double det;
  HepMatrix mt(*this);
  int i = mt.dfact_matrix(det, ir);
  if(i==0) return det;
  return 0;
}

double HepMatrix::trace() const {
   double t = 0.0;
   for (mcIter d = m.begin(); d < m.end(); d += (ncol+1) )
      t += *d;
   return t;
}

}  // namespace CLHEP
