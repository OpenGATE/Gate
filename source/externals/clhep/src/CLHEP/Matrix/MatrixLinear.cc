// -*- C++ -*-
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
// This is the definition of special linear algebra functions for the
// HepMatrix class.
//
// Many of these algorithms are taken from Golub and Van Loan.
//

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"

namespace CLHEP {

static inline int sign(double x) { return (x>0 ? 1: -1);}

/* -----------------------------------------------------------------------

   The following contains stuff to try to do basic things with matrixes.
   The functions are:

   back_solve         - Solves R*x = b where R is upper triangular.
                        Also has a variation that solves a number of equations
			of this form in one step, where b is a matrix with
			each column a different vector. See also solve.
   col_givens         - Does a column Givens update.
   col_house          - Does a column Householder update.
   condition          - Find the conditon number of a symmetric matrix.
   diag_step          - Implicit symmetric QR step with Wilkinson Shift
   diagonalize        - Diagonalize a symmetric matrix.
   givens             - algorithm 5.1.5 in Golub and Van Loan
   house              - Returns a Householder vector to zero elements.
   house_with_update  - Finds and does Householder reflection on matrix.
   qr_inverse         - Finds the inverse of a matrix.  Note, often what you
                        really want is solve or backsolve, they can be much
			quicker than inverse in many calculations.
   min_line_dist      - Takes a set of lines and returns the point nearest to
                        all the lines in the least squares sense.
   qr_decomp          - Does a QR decomposition of a matrix.
   row_givens         - Does a row Givens update.
   row_house          - Does a row Householder update.
   qr_solve           - Works like backsolve, except matrix does not need
                        to be upper triangular. For nonsquare matrix, it
			solves in the least square sense.
   tridiagonal        - Does a Householder tridiagonalization of a symmetric
                        matrix.
   ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
   back_solve Version: 1.00 Date Last Changed: 2/9/93

   This solves the system R*x=b where R is upper triangular.  Since b
   is most likely not interesting after this step, x is returned in b.
   This is algorithm 3.1.2 in Golub & Van Loan
   ----------------------------------------------------------------------- */

void back_solve(const HepMatrix &R, HepVector *b)
{
   (*b)(b->num_row()) /= R(b->num_row(),b->num_row());
   int n = R.num_col();
   int nb = b->num_row();
   HepMatrix::mIter br = b->m.begin() + b->num_row() - 2;
   HepMatrix::mcIter Rrr = R.m.begin() + (nb-2) * (n+1);
   for (int r=b->num_row()-1;r>=1;--r) {
      HepMatrix::mIter bc = br+1;
      HepMatrix::mcIter Rrc = Rrr+1;
      for (int c=r+1;c<=b->num_row();c++) {
	 (*br)-=(*(Rrc++))*(*(bc++));
      }
      (*br)/=(*Rrr);
      if(r>1) {
	 br--;
	 Rrr -= n+1;
      }
   }
}

/* -----------------------------------------------------------------------
   Variation that solves a set of equations R*x=b in one step, where b
   is a Matrix with columns each a different vector.  Solution is again
   returned in b.
   ----------------------------------------------------------------------- */

void back_solve(const HepMatrix &R, HepMatrix *b)
{
   int n = R.num_col();
   int nb = b->num_row();
   int nc = b->num_col();
   HepMatrix::mIter bbi = b->m.begin() + (nb - 2) * nc;
   for (int i=1;i<=b->num_col();i++) {
      (*b)(b->num_row(),i) /= R(b->num_row(),b->num_row());
      HepMatrix::mcIter Rrr = R.m.begin() + (nb-2) * (n+1);
      HepMatrix::mIter bri = bbi;
      for (int r=b->num_row()-1;r>=1;--r) {
	 HepMatrix::mIter bci = bri + nc;
	 HepMatrix::mcIter Rrc = Rrr+1;
	 for (int c=r+1;c<=b->num_row();c++) {
	    (*bri)-= (*(Rrc++)) * (*bci);
	    if(c<b->num_row()) bci += nc;
	 }
	 (*bri)/= (*Rrr);
	 if(r>1) {
	    Rrr -= (n+1);
	    bri -= nc;
	 }
      }
      bbi++;
   }
}

/* -----------------------------------------------------------------------
   col_givens Version: 1.00 Date Last Changed: 1/28/93

   This does the same thing as row_givens, except it works for columns.
   It replaces A with A*G.
   ----------------------------------------------------------------------- */

void col_givens(HepMatrix *A, double c,double ds,
		int k1, int k2, int row_min, int row_max) {
   if (row_max<=0) row_max = A->num_row();
   int n = A->num_col();
   HepMatrix::mIter Ajk1 = A->m.begin() + (row_min - 1) * n + k1 - 1;
   HepMatrix::mIter Ajk2 = A->m.begin() + (row_min - 1) * n + k2 - 1;
   for (int j=row_min;j<=row_max;j++) {
      double tau1=(*Ajk1); double tau2=(*Ajk2);
      (*Ajk1)=c*tau1-ds*tau2;(*Ajk2)=ds*tau1+c*tau2;
      if(j<row_max) {
	 Ajk1 += n;
	 Ajk2 += n;
      }
   }
}

/* -----------------------------------------------------------------------
   col_house Version: 1.00 Date Last Changed: 1/28/93

   This replaces A with A*P where P=I-2v*v.T/(v.T*v).  If row and col are
   not one, then it only acts on the subpart of A, from A(row,col) to 
   A(A.num_row(),A.num_row()).  For use with house, can pass V as a matrix.
   Then row_start and col_start describe where the vector lies.  It is
   assumed to run from V(row_start,col_start) to 
   V(row_start+A.num_row()-row,col_start).
   Since typically row_house comes after house, and v.normsq is calculated
   in house, it is passed as a arguement.  Also supplied without vnormsq.
   This does a column Householder update.
   ----------------------------------------------------------------------- */

void col_house(HepMatrix *a,const HepMatrix &v,double vnormsq,
	       int row, int col, int row_start,int col_start) {
   double beta=-2/vnormsq;

   // This is a fast way of calculating w=beta*A.sub(row,n,col,n)*v

   HepVector w(a->num_col()-col+1,0);
/* not tested */
   HepMatrix::mIter wptr = w.m.begin();
   int n = a->num_col();
   int nv = v.num_col();
   HepMatrix::mIter acrb = a->m.begin() + (col-1) * n + (row-1);
   int c;
   for (c=col;c<=a->num_col();c++) {
      HepMatrix::mcIter vp = v.m.begin() + (row_start-1) * nv + (col_start-1);
      HepMatrix::mcIter acr = acrb;
      for (int r=row;r<=a->num_row();r++) {
	 (*wptr)+=(*(acr++))*(*vp);
	 vp += nv;
      }
      wptr++;
      if(c<a->num_col()) acrb += n;
   }
   w*=beta;
  
   // Fast way of calculating A.sub=A.sub+w*v.T()

   HepMatrix::mIter arcb = a->m.begin() + (row-1) * n + col-1;
   wptr = w.m.begin();
   for (int r=row; r<=a->num_row();r++) {
      HepMatrix::mIter arc = arcb;
      HepMatrix::mcIter vp = v.m.begin() + (row_start-1) * nv + col_start;
      for (c=col;c<=a->num_col();c++) {
	 (*(arc++))+=(*vp)*(*wptr);
	 vp += nv;
      }
      wptr++;
      if(r<a->num_row()) arcb += n;
   }
}

/* -----------------------------------------------------------------------
   condition Version: 1.00 Date Last Changed: 1/28/93

   This finds the condition number of SymMatrix.
   ----------------------------------------------------------------------- */

double condition(const HepSymMatrix &hm)
{
   HepSymMatrix mcopy=hm;
   diagonalize(&mcopy);
   double max,min;
   max=min=fabs(mcopy(1,1));

   int n = mcopy.num_row();
   HepMatrix::mIter mii = mcopy.m.begin() + 2;
   for (int i=2; i<=n; i++) {
      if (max<fabs(*mii)) max=fabs(*mii);
      if (min>fabs(*mii)) min=fabs(*mii);
      if(i<n) mii += i+1;
   } 
   return max/min;
}

/* -----------------------------------------------------------------------
   diag_step Version: 1.00 Date Last Changed: 1/28/93

   This does a Implicit symmetric QR step with Wilkinson Shift.  See 
   algorithm 8.2.2 in Golub and Van Loan. begin and end mark the submatrix
   of t to do the QR step on, the matrix runs from t(begin,begin) to 
   t(end,end).  Can include Matrix *U to be updated so that told = U*t*U.T();
   ----------------------------------------------------------------------- */

void diag_step(HepSymMatrix *t,int begin,int end)
{
   double d=(t->fast(end-1,end-1)-t->fast(end,end))/2;
   double mu=t->fast(end,end)-t->fast(end,end-1)*t->fast(end,end-1)/
	 (d+sign(d)*sqrt(d*d+ t->fast(end,end-1)*t->fast(end,end-1)));
   double x=t->fast(begin,begin)-mu;
   double z=t->fast(begin+1,begin);
   HepMatrix::mIter tkk = t->m.begin() + (begin+2)*(begin-1)/2;
   HepMatrix::mIter tkp1k = tkk + begin;
   HepMatrix::mIter tkp2k = tkk + 2 * begin + 1;
   for (int k=begin;k<=end-1;k++) {
      double c,ds;
      givens(x,z,&c,&ds);

      // This is the result of G.T*t*G, making use of the special structure
      // of t and G. Note that since t is symmetric, only the lower half
      // needs to be updated.  Equations were gotten from maple.

      if (k!=begin)
      {
	 *(tkk-1)= *(tkk-1)*c-(*(tkp1k-1))*ds;
	 *(tkp1k-1)=0;
      }
      double ap=(*tkk);
      double bp=(*tkp1k);
      double aq=(*tkp1k+1);
      (*tkk)=ap*c*c-2*c*bp*ds+aq*ds*ds;
      (*tkp1k)=c*ap*ds+bp*c*c-bp*ds*ds-ds*aq*c;
      (*(tkp1k+1))=ap*ds*ds+2*c*bp*ds+aq*c*c;
      if (k<end-1)
      {
	 double bq=(*(tkp2k+1));
	 (*tkp2k)=-bq*ds;
	 (*(tkp2k+1))=bq*c;
	 x=(*tkp1k);
	 z=(*tkp2k);
	 tkk += k+1;
	 tkp1k += k+2;
      }
      if(k<end-2) tkp2k += k+3;
   }
}

void diag_step(HepSymMatrix *t,HepMatrix *u,int begin,int end)
{
   double d=(t->fast(end-1,end-1)-t->fast(end,end))/2;
   double mu=t->fast(end,end)-t->fast(end,end-1)*t->fast(end,end-1)/
	 (d+sign(d)*sqrt(d*d+ t->fast(end,end-1)*t->fast(end,end-1)));
   double x=t->fast(begin,begin)-mu;
   double z=t->fast(begin+1,begin);
   HepMatrix::mIter tkk = t->m.begin() + (begin+2)*(begin-1)/2;
   HepMatrix::mIter tkp1k = tkk + begin;
   HepMatrix::mIter tkp2k = tkk + 2 * begin + 1;
   for (int k=begin;k<=end-1;k++) {
      double c,ds;
      givens(x,z,&c,&ds);
      col_givens(u,c,ds,k,k+1);

      // This is the result of G.T*t*G, making use of the special structure
      // of t and G. Note that since t is symmetric, only the lower half
      // needs to be updated.  Equations were gotten from maple.

      if (k!=begin) {
	 *(tkk-1)= (*(tkk-1))*c-(*(tkp1k-1))*ds;
	 *(tkp1k-1)=0;
      }
      double ap=(*tkk);
      double bp=(*tkp1k);
      double aq=(*(tkp1k+1));
      (*tkk)=ap*c*c-2*c*bp*ds+aq*ds*ds;
      (*tkp1k)=c*ap*ds+bp*c*c-bp*ds*ds-ds*aq*c;
      (*(tkp1k+1))=ap*ds*ds+2*c*bp*ds+aq*c*c;
      if (k<end-1) {
	 double bq=(*(tkp2k+1));
	 (*tkp2k)=-bq*ds;
	 (*(tkp2k+1))=bq*c;
	 x=(*tkp1k);
	 z=(*(tkp2k));
	 tkk += k+1;
	 tkp1k += k+2;
      }
      if(k<end-2) tkp2k += k+3;
   }
}

/* -----------------------------------------------------------------------
   diagonalize Version: 1.00 Date Last Changed: 1/28/93

   This subroutine diagonalizes a symmatrix using algorithm 8.2.3 in Golub &
   Van Loan.  It returns the matrix U so that sold = U*sdiag*U.T
   ----------------------------------------------------------------------- */
HepMatrix diagonalize(HepSymMatrix *hms)
{
   const double tolerance = 1e-12;
   HepMatrix u= tridiagonal(hms);
   int begin=1;
   int end=hms->num_row();
   while(begin!=end)
   {
      HepMatrix::mIter sii = hms->m.begin() + (begin+2)*(begin-1)/2;
      HepMatrix::mIter sip1i = sii + begin;
      for (int i=begin;i<=end-1;i++) {
	 if (fabs(*sip1i)<=
	    tolerance*(fabs(*sii)+fabs(*(sip1i+1)))) {
	    (*sip1i)=0;
	 }
	 if(i<end-1) {
	    sii += i+1;
	    sip1i += i+2;
	 }
      }
      while(begin<end && hms->fast(begin+1,begin) ==0) begin++;
      while(end>begin && hms->fast(end,end-1) ==0) end--;
      if (begin!=end)
	 diag_step(hms,&u,begin,end);
   }
   return u;
}

/* -----------------------------------------------------------------------
   house Version: 1.00 Date Last Changed: 1/28/93

   This return a Householder Vector to zero the elements in column col, 
   from row+1 to a.num_row().
   ----------------------------------------------------------------------- */
     
HepVector house(const HepSymMatrix &a,int row,int col)
{
   HepVector v(a.num_row()-row+1);
/* not tested */
   HepMatrix::mIter vp = v.m.begin();
   HepMatrix::mcIter aci = a.m.begin() + col * (col - 1) / 2 + row - 1;
   int i;
   for (i=row;i<=col;i++) {
      (*(vp++))=(*(aci++));
   }
   for (;i<=a.num_row();i++) {
      (*(vp++))=(*aci);
      aci += i;
   }
   v(1)+=sign(a(row,col))*v.norm();
   return v;
}

HepVector house(const HepMatrix &a,int row,int col)
{
   HepVector v(a.num_row()-row+1);
/* not tested */
   int n = a.num_col();
   HepMatrix::mcIter aic = a.m.begin() + (row - 1) * n + (col - 1) ;
   HepMatrix::mIter vp = v.m.begin();
   for (int i=row;i<=a.num_row();i++) {
      (*(vp++))=(*aic);
      aic += n;
   }
   v(1)+=sign(a(row,col))*v.norm();
   return v;
}

/* -----------------------------------------------------------------------
   house_with_update Version: 1.00 Date Last Changed: 1/28/93

   This returns the householder vector as in house, and it also changes
   A to be PA. (It is slightly more efficient than calling house, followed
   by calling row_house).  If you include the optional Matrix *house_vector,
   then the householder vector is stored in house_vector(row,col) to
   house_vector(a->num_row(),col).
   ----------------------------------------------------------------------- */

void house_with_update(HepMatrix *a,int row,int col)
{
   HepVector v(a->num_row()-row+1);
/* not tested */
   HepMatrix::mIter vp = v.m.begin();
   int n = a->num_col();
   HepMatrix::mIter arc = a->m.begin() + (row-1) * n + (col-1);
   int r;
   for (r=row;r<=a->num_row();r++) {
      (*(vp++))=(*arc);
      if(r<a->num_row()) arc += n;
   }
   double normsq=v.normsq();
   double norm=sqrt(normsq);
   normsq-=v(1)*v(1);
   v(1)+=sign((*a)(row,col))*norm;
   normsq+=v(1)*v(1);
   (*a)(row,col)=-sign((*a)(row,col))*norm;
   if (row<a->num_row()) {
      arc = a->m.begin() + row * n + (col-1);
      for (r=row+1;r<=a->num_row();r++) {
	 (*arc)=0;
	 if(r<a->num_row()) arc += n;
      }
      row_house(a,v,normsq,row,col+1);
   }
}

void house_with_update(HepMatrix *a,HepMatrix *v,int row,int col)
{
   double normsq=0;
   int nv = v->num_col();
   int na = a->num_col();
   HepMatrix::mIter vrc = v->m.begin() + (row-1) * nv + (col-1);
   HepMatrix::mIter arc = a->m.begin() + (row-1) * na + (col-1);
   int r;
   for (r=row;r<=a->num_row();r++) {
      (*vrc)=(*arc);
      normsq+=(*vrc)*(*vrc);
      if(r<a->num_row()) {
	 vrc += nv;
	 arc += na;
      }
   }
   double norm=sqrt(normsq);
   vrc = v->m.begin() + (row-1) * nv + (col-1);
   normsq-=(*vrc)*(*vrc);
   (*vrc)+=sign((*a)(row,col))*norm;
   normsq+=(*vrc)*(*vrc);
   (*a)(row,col)=-sign((*a)(row,col))*norm;
   if (row<a->num_row()) {
      arc = a->m.begin() + row * na + (col-1);
      for (r=row+1;r<=a->num_row();r++) {
	 (*arc)=0;
	 if(r<a->num_row()) arc += na;
      }
      row_house(a,*v,normsq,row,col+1,row,col);
   }
}
/* -----------------------------------------------------------------------
   house_with_update2 Version: 1.00 Date Last Changed: 1/28/93

   This is a variation of house_with_update for use with tridiagonalization.
   It only updates column number col in a SymMatrix.
   ----------------------------------------------------------------------- */

void house_with_update2(HepSymMatrix *a,HepMatrix *v,int row,int col)
{
   double normsq=0;
   int nv = v->num_col();
   HepMatrix::mIter vrc = v->m.begin() + (row-1) * nv + (col-1);
   HepMatrix::mIter arc = a->m.begin() + (row-1) * row / 2 + (col-1);
   int r;
   for (r=row;r<=a->num_row();r++)
   {
      (*vrc)=(*arc);
      normsq+=(*vrc)*(*vrc);
      if(r<a->num_row()) {
	 arc += r;
	 vrc += nv;
      }
   }
   double norm=sqrt(normsq);
   vrc = v->m.begin() + (row-1) * nv + (col-1);
   arc = a->m.begin() + (row-1) * row / 2 + (col-1);
   (*vrc)+=sign(*arc)*norm;
   (*arc)=-sign(*arc)*norm;
   arc += row;
   for (r=row+1;r<=a->num_row();r++) {
      (*arc)=0;
      if(r<a->num_row()) arc += r;
   }
}

/* -----------------------------------------------------------------------
   inverse Version: 1.00 Date Last Changed: 2/9/93
   
   This calculates the inverse of a square matrix.  Note that this is 
   often not what you really want to do.  Often, you really want solve or
   backsolve, which can be faster at calculating (e.g. you want the inverse
   to calculate A^-1*c where c is a vector.  Then this is just solve(A,c))
   
   If A is not need after the calculation, you can call this with *A.  A will
   be destroyed, but the inverse will be calculated quicker.
   ----------------------------------------------------------------------- */

HepMatrix qr_inverse(const HepMatrix &A)
{
   HepMatrix Atemp=A;
   return qr_inverse(&Atemp);
}

HepMatrix qr_inverse(HepMatrix *A)
{
   if (A->num_row()!=A->num_col()) {
      HepGenMatrix::error("qr_inverse: The matrix is not square.");
   }
   HepMatrix QT=qr_decomp(A).T();
   back_solve(*A,&QT);
   return QT;
}

/* -----------------------------------------------------------------------
   Version: 1.00 Date Last Changed: 5/26/93

   This takes a set of lines described by Xi=Ai*t+Bi and finds the point P
   that is closest to the lines in the least squares sense.  n is the
   number of lines, and A and B are pointers to arrays of the Vectors Ai
   and Bi.  The array runs from 0 to n-1.
   ----------------------------------------------------------------------- */

HepVector min_line_dist(const HepVector *const A, const HepVector *const B,
			 int n)
{
   // For (P-(A t +B))^2 minimum, we have tmin=dot(A,P-B)/A.normsq().  So
   // To minimize distance, we want sum_i (P-(Ai tmini +Bi))^2 minimum.  This
   // is solved by having 
   // (sum_i k Ai*Ai.T +1) P - (sum_i k dot(Ai,Bi) Ai + Bi) = 0
   // where k=1-2/Ai.normsq
   const double small = 1e-10;  // Small number
   HepSymMatrix C(3,0),I(3,1);
   HepVector D(3,0);
   double t;
   for (int i=0;i<n;i++)
   {
      if (fabs(t=A[i].normsq())<small) {
	 C += I;
	 D += B[i];
      } else {
	 C += vT_times_v(A[i])*(1-2/t)+I;
	 D += dot(A[i],B[i])*(1-2/t)*A[i]+B[i];
      }
   }
   return qr_solve(C,D);
}

/* -----------------------------------------------------------------------
   qr_decomp Version: 1.00 Date Last Changed: 1/28/93

   This does a Householder QR decomposition of the passed matrix.  The
   matrix does not have to be square, but the number of rows needs to be
   >= number of columns.   If A is a mxn matrix, Q is mxn and R is nxn.
   R is returned in the upper left part of A.  qr_decomp with a second
   matrix changes A to R, and returns a set of householder vectors.

   Algorithm is from Golub and Van Loan 5.15.
   ----------------------------------------------------------------------- */

HepMatrix qr_decomp(HepMatrix *A)
{
   HepMatrix hsm(A->num_row(),A->num_col());
   qr_decomp(A,&hsm);
   // Try to make Q diagonal
   //  HepMatrix Q(A->num_row(),A->num_col(),1);
   HepMatrix Q(A->num_row(),A->num_row(),1);
   for (int j=hsm.num_col();j>=1;--j)
      row_house(&Q,hsm,j,j,j,j);
   return Q;
}

/* -----------------------------------------------------------------------
   row_givens Version: 1.00 Date Last Changed: 1/28/93

   This algorithm performs a Givens rotation on the left.  Given c and s
   and k1 and k2, this is like forming G= identity except for row k1 and 
   k2 where G(k1,k1)=c, G(k1,k2)=s, G(k2,k1)=-s, G(k2,k2)=c.  It replaces
   A with G.T()*A.  A variation allows you to express col_min and col_max,
   and then only the submatrix of A from (1,col_min) to (num_row(),col_max)
   are updated.  This is algorithm 5.1.6 in Golub and Van Loan.
   ----------------------------------------------------------------------- */

void row_givens(HepMatrix *A, double c,double ds,
		int k1, int k2, int col_min, int col_max) {
   /* not tested */
   if (col_max==0) col_max = A->num_col();
   int n = A->num_col();
   HepMatrix::mIter Ak1j = A->m.begin() + (k1-1) * n + (col_min-1);
   HepMatrix::mIter Ak2j = A->m.begin() + (k2-1) * n + (col_min-1);
   for (int j=col_min;j<=col_max;j++) {
      double tau1=(*Ak1j); double tau2=(*Ak2j);
      (*(Ak1j++))=c*tau1-ds*tau2;(*(Ak2j++))=ds*tau1+c*tau2;
   }
}

/* -----------------------------------------------------------------------
   row_house Version: 1.00 Date Last Changed: 1/28/93

   This replaces A with P*A where P=I-2v*v.T/(v.T*v).  If row and col are
   not one, then it only acts on the subpart of A, from A(row,col) to 
   A(A.num_row(),A.num_row()).  For use with house, can pass V as a matrix.
   Then row_start and col_start describe where the vector lies.  It is
   assumed to run from V(row_start,col_start) to 
   V(row_start+A.num_row()-row,col_start).
   Since typically row_house comes after house, and v.normsq is calculated
   in house, it is passed as a arguement.  Also supplied without vnormsq.
   ----------------------------------------------------------------------- */

void row_house(HepMatrix *a,const HepVector &v,double vnormsq,
	       int row, int col) {
   double beta=-2/vnormsq;

   // This is a fast way of calculating w=beta*A.sub(row,n,col,n).T()*v

   HepVector w(a->num_col()-col+1,0);
/* not tested */
   int na = a->num_col();
   HepMatrix::mIter wptr = w.m.begin();
   HepMatrix::mIter arcb = a->m.begin() + (row-1) * na + (col-1);
   int c;
   for (c=col;c<=a->num_col();c++) {
      HepMatrix::mcIter vp = v.m.begin();
      HepMatrix::mIter arc = arcb;
      for (int r=row;r<=a->num_row();r++) {
	 (*wptr)+=(*arc)*(*(vp++));
	 if(r<a->num_row()) arc += na;
      }
      wptr++;
      arcb++;
   }
   w*=beta;
  
   // Fast way of calculating A.sub=A.sub+v*w.T()

   arcb = a->m.begin() + (row-1) * na + (col-1);
   HepMatrix::mcIter vp = v.m.begin();
   for (int r=row; r<=a->num_row();r++) {
      HepMatrix::mIter wptr2 = w.m.begin();
      HepMatrix::mIter arc = arcb;
      for (c=col;c<=a->num_col();c++) {
	 (*(arc++))+=(*vp)*(*(wptr2++));
      }
      vp++;
      if(r<a->num_row()) arcb += na;
   }
}

void row_house(HepMatrix *a,const HepMatrix &v,double vnormsq,
	       int row, int col, int row_start, int col_start) {
   double beta=-2/vnormsq;

   // This is a fast way of calculating w=beta*A.sub(row,n,col,n).T()*v
   HepVector w(a->num_col()-col+1,0);
   int na = a->num_col();
   int nv = v.num_col();
   HepMatrix::mIter wptr = w.m.begin();
   HepMatrix::mIter arcb = a->m.begin() + (row-1) * na + (col-1);
   HepMatrix::mcIter vpcb = v.m.begin() + (row_start-1) * nv + (col_start-1);
   int c;
   for (c=col;c<=a->num_col();c++) {
      HepMatrix::mIter arc = arcb;
      HepMatrix::mcIter vpc = vpcb;
      for (int r=row;r<=a->num_row();r++) {
	 (*wptr)+=(*arc)*(*vpc);
         if(r<a->num_row()) {
	   arc += na;
	   vpc += nv;
	 }
      }
      wptr++;
      arcb++;
   }
   w*=beta;

   arcb = a->m.begin() + (row-1) * na + (col-1);
   HepMatrix::mcIter vpc = v.m.begin() + (row_start-1) * nv + (col_start-1);
   for (int r=row; r<=a->num_row();r++) {
      HepMatrix::mIter arc = arcb;
      HepMatrix::mIter wptr2 = w.m.begin();
      for (c=col;c<=a->num_col();c++) {
	 (*(arc++))+=(*vpc)*(*(wptr2++));
      }
      if(r<a->num_row()) {
	arcb += na;
	vpc += nv;
      }
   }
}

/* -----------------------------------------------------------------------
   solve Version: 1.00 Date Last Changed: 2/9/93

   This solves the system A*x=b where A is not upper triangular, but it
   must have full rank. If A is not square, then this is solved in the least
   squares sense. Has a variation that works for b a matrix with each column 
   being a different vector.  If A is not needed after this call, you can 
   call solve with *A.  This will destroy A, but it will run faster.
   ----------------------------------------------------------------------- */

HepVector qr_solve(const HepMatrix &A, const HepVector &b)
{
   HepMatrix temp = A;
   return qr_solve(&temp,b);
}

HepVector qr_solve(HepMatrix *A,const HepVector &b)
{
   HepMatrix Q=qr_decomp(A);
   // Quick way to to Q.T*b.
   HepVector b2(Q.num_col(),0);
   HepMatrix::mIter b2r = b2.m.begin();
   HepMatrix::mIter Qr = Q.m.begin();
   int n = Q.num_col();
   for (int r=1;r<=b2.num_row();r++) {
      HepMatrix::mcIter bc = b.m.begin();
      HepMatrix::mIter Qcr = Qr;
      for (int c=1;c<=b.num_row();c++) {
	 *b2r += (*Qcr) * (*(bc++));
	 if(c<b.num_row()) Qcr += n;
      }
      b2r++;
      Qr++;
   }
   back_solve(*A,&b2);
   return b2;
}

HepMatrix qr_solve(const HepMatrix &A, const HepMatrix &b)
{
   HepMatrix temp = A;
   return qr_solve(&temp,b);
}

HepMatrix qr_solve(HepMatrix *A,const HepMatrix &b)
{
   HepMatrix Q=qr_decomp(A);
   // Quick way to to Q.T*b.
   HepMatrix b2(Q.num_col(),b.num_col(),0);
   int nb = b.num_col();
   int nq = Q.num_col();
   HepMatrix::mcIter b1i = b.m.begin();
   HepMatrix::mIter b21i = b2.m.begin();
   for (int i=1;i<=b.num_col();i++) {
      HepMatrix::mIter Q1r = Q.m.begin();
      HepMatrix::mIter b2ri = b21i;
      for (int r=1;r<=b2.num_row();r++) {
	 HepMatrix::mIter Qcr = Q1r;
	 HepMatrix::mcIter bci = b1i;
	 for (int c=1;c<=b.num_row();c++) {
	    *b2ri += (*Qcr) * (*bci);
	    if(c<b.num_row()) {
	       Qcr += nq;
	       bci += nb;
	    }
	 }
	 Q1r++;
	 if(r<b2.num_row()) b2ri += nb;
      }
      b1i++;
      b21i++;
   }
   back_solve(*A,&b2);
   return b2;
}

/* -----------------------------------------------------------------------
   tridiagonal Version: 1.00 Date Last Changed: 1/28/93

   This does a Householder tridiagonalization.  It takes a symmetric matrix
   A and changes it to A=U*T*U.T.
   ----------------------------------------------------------------------- */

void tridiagonal(HepSymMatrix *a,HepMatrix *hsm)
{
   int nh = hsm->num_col();
   for (int k=1;k<=a->num_col()-2;k++) {
      
      // If this row is already in tridiagonal form, we can skip the
      // transformation.

      double scale=0;
      HepMatrix::mIter ajk = a->m.begin() + k * (k+5) / 2;
      int j;
      for (j=k+2;j<=a->num_row();j++) {
	 scale+=fabs(*ajk);
	 if(j<a->num_row()) ajk += j;
      }
      if (scale ==0) {
	 HepMatrix::mIter hsmjkp = hsm->m.begin() + k * (nh+1) - 1;
	 for (j=k+1;j<=hsm->num_row();j++) {
	    *hsmjkp = 0;
	    if(j<hsm->num_row()) hsmjkp += nh;
	 }
      } else {
	 house_with_update2(a,hsm,k+1,k);
	 double normsq=0;
	 HepMatrix::mIter hsmrptrkp = hsm->m.begin() + k * (nh+1) - 1;
	 int rptr;
	 for (rptr=k+1;rptr<=hsm->num_row();rptr++) {
	    normsq+=(*hsmrptrkp)*(*hsmrptrkp);
	    if(rptr<hsm->num_row()) hsmrptrkp += nh;
	 }
	 HepVector p(a->num_row()-k,0);
	 rptr=k+1;
	 HepMatrix::mIter pr = p.m.begin();
	 int r;
	 for (r=1;r<=p.num_row();r++) {
	    HepMatrix::mIter hsmcptrkp = hsm->m.begin() + k * (nh+1) - 1;
	    int cptr;
	    for (cptr=k+1;cptr<=rptr;cptr++) {
	       (*pr)+=a->fast(rptr,cptr)*(*hsmcptrkp);
	       if(cptr<a->num_col()) hsmcptrkp += nh;
	    }
	    for (;cptr<=a->num_col();cptr++) {
	       (*pr)+=a->fast(cptr,rptr)*(*hsmcptrkp);
	       if(cptr<a->num_col()) hsmcptrkp += nh;
	    }
	    (*pr)*=2/normsq;
	    rptr++;
	    pr++;
	 }
	 double pdotv=0;
	 pr = p.m.begin();
	 hsmrptrkp = hsm->m.begin() + k * (nh+1) - 1;
	 for (r=1;r<=p.num_row();r++) {
	    pdotv+=(*(pr++))*(*hsmrptrkp);
	    if(r<p.num_row()) hsmrptrkp += nh;
	 }
	 pr = p.m.begin();
	 hsmrptrkp = hsm->m.begin() + k * (nh+1) - 1;
	 for (r=1;r<=p.num_row();r++) {
	    (*(pr++))-=pdotv*(*hsmrptrkp)/normsq;
	    if(r<p.num_row()) hsmrptrkp += nh;
	 }
	 rptr=k+1;
	 pr = p.m.begin();
	 hsmrptrkp = hsm->m.begin() + k * (nh+1) - 1;
	 for (r=1;r<=p.num_row();r++) {
	    int cptr=k+1;
	    HepMatrix::mIter mpc = p.m.begin();
	    HepMatrix::mIter hsmcptrkp = hsm->m.begin() + k * (nh+1) - 1;
	    for (int c=1;c<=r;c++) {
	       a->fast(rptr,cptr)-= (*hsmrptrkp)*(*(mpc++))+(*pr)*(*hsmcptrkp);
	       cptr++;
	       if(c<r) hsmcptrkp += nh;
	    }
	    pr++;
	    rptr++;
	    if(r<p.num_row()) hsmrptrkp += nh;
	 }
      }
   }
}

HepMatrix tridiagonal(HepSymMatrix *a)
{
   HepMatrix U(a->num_row(),a->num_col(),1);
   if (a->num_col()>2)
   {
      HepMatrix hsm(a->num_col(),a->num_col()-2,0);
      tridiagonal(a,&hsm);
      for (int j=hsm.num_col();j>=1;--j) {
	 row_house(&U,hsm,j,j,j,j);
      }
   }
   return U;
}

void col_house(HepMatrix *a,const HepMatrix &v,int row,int col,
	       int row_start,int col_start)
{
   double normsq=0;
   for (int i=row_start;i<=row_start+a->num_row()-row;i++)
      normsq+=v(i,col)*v(i,col);
   col_house(a,v,normsq,row,col,row_start,col_start);
}

void givens(double a, double b, double *c, double *ds) 
{
   if (b ==0) { *c=1; *ds=0; }
   else {
      if (fabs(b)>fabs(a)) {
	 double tau=-a/b;
	 *ds=1/sqrt(1+tau*tau);
	 *c=(*ds)*tau;
      } else {
	 double tau=-b/a;
	 *c=1/sqrt(1+tau*tau);
	 *ds=(*c)*tau;
      }
   }
}

void qr_decomp(HepMatrix *A,HepMatrix *hsm)
{
   for (int i=1;i<=A->num_col();i++)
      house_with_update(A,hsm,i,i);
}

void row_house(HepMatrix *a,const HepMatrix &v,int row,int col,
	       int row_start,int col_start)
{
   double normsq=0;
   int end = row_start+a->num_row()-row;
   for (int i=row_start; i<=end; i++)
      normsq += v(i,col)*v(i,col);
   // If v is 0, then we can skip doing row_house.  
   if (normsq !=0)
      row_house(a,v,normsq,row,col,row_start,col_start);
}

}  // namespace CLHEP
