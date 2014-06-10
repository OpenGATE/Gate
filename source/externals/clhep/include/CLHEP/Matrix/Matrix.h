// -*- C++ -*-
// CLASSDOC OFF
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
// 
// This is the definition of the HepMatrix class.
// 
// This software written by Nobu Katayama and Mike Smyth, Cornell University.
//
//
// .SS Usage
// The Matrix class does all the obvious things, in all the obvious ways.
// You declare a Matrix by saying
//
// .ft B
//       HepMatrix hm1(n, m);
//
//  To declare a Matrix as a copy of a Matrix hm2, say
//
// .ft B
//       HepMatrix hm1(hm2);
// or
// .ft B
//       HepMatrix hm1 = hm2;
// 
// You can declare initilizations of a Matrix, by giving it a third
// integer argument, either 0 or 1. 0 means initialize to 0, one means
// the identity matrix. If you do not give a third argument the memory
// is not initialized.
//
// .ft B
//       HepMatrix hm1(n, m, 1);
//
// ./"This code has been written by Mike Smyth, and the algorithms used are
// ./"described in the thesis "A Tracking Library for a Silicon Vertex Detector"
// ./"(Mike Smyth, Cornell University, June 1993).
// ./"This is file contains C++ stuff for doing things with Matrices.
// ./"To turn on bound checking, define MATRIX_BOUND_CHECK before including
// ./"this file.
// 
//  To find the number of rows or number of columns, say 
//
// .ft B
// nr = hm1.num_row();
//
// or
//
// .ft B
// nc = hm1.num_col();
//
// You can print a Matrix by
//
// .ft B
// cout << hm1;
//
//  You can add,
//  subtract, and multiply two Matrices.  You can multiply a Matrix by a
//  scalar, on the left or the right.  +=, *=, -= act in the obvious way.
//  hm1 *= hm2 is as hm1 = hm1*hm2. You can also divide by a scalar on the
//  right, or use /= to do the same thing.  
// 
//  You can read or write a Matrix element by saying
//
// .ft B
//  m(row, col) = blah. (1 <= row <= num_row(), 1 <= col <= num_col())
//
// .ft B
//  blah = m(row, col) ...
//
//  m(row, col) is inline, and by default does not do bound checking. 
//  If bound checking is desired, say #define MATRIX_BOUND_CHECK before
//  including matrix.h.
// 
//  You can also access the element using C subscripting operator []
//
// .ft B
//  m[row][col] = blah. (0 <= row < num_row(), 0 <= col < num_col())
//
// .ft B
//  blah = m[row][col] ...
//
//  m[row][col] is inline, and by default does not do bound checking. 
//  If bound checking is desired, say #define MATRIX_BOUND_CHECK before
//  including matrix.h. (Notice the difference in bases in two access
//  methods.) 
//
// .SS Comments on precision
//
//  The user would normally use "Matrix" class whose precision is the same
//  as the other classes of CLHEP (ThreeVec, for example). However, he/she
//  can explicitly choose Matrix (double) or MatrixF (float) if he/she wishes.
//  In the former case, include "Matrix.h." In the latter case, include either
//  "Matrix.h," or "MatrixF.h," or both. The only operators that can talk
//  to each other are the copy constructor and assignment operator.
//
// .ft B
//  Matrix d(3,4,HEP_MATRIX_IDENTITY);
//
// .ft B
//  MatrixF f;
//
// .ft B
//  f = d;
//
// .ft B
//  MatrixF g(d);
//
//  will convert from one to the other.
//
// .SS Other functions
//
// .ft B
//  mt = m.T();
//
//  returns the transpose of m. 
//
// .ft B
//  ms = hm2.sub(row_min, row_max, col_min, col_max);
//
//  returns the subMatrix.
//  hm2(row_min:row_max, col_min:col_max) in matlab terminology.
//  If instead you say
//
// .ft B
//  hm2.sub(row, col, hm1);
//
//  then the subMatrix
//  hm2(row:row+hm1.num_row()-1, col:col+hm1.num_col()-1) is replaced with hm1.
//
// .ft B
// md = dsum(hm1,hm2);
//
// returns the direct sum of the two matrices.
//
// Operations that do not have to be member functions or friends are listed
// towards the end of this man page.
//
//
// To invert a matrix, say
//
// .ft B
// minv = m.inverse(ierr);
//
// ierr will be non-zero if the matrix is singular.
//
// If you can overwrite the matrix, you can use the invert function to avoid
// two extra copies. Use
//
// .ft B
// m.invert(ierr);
//
// Many basic linear algebra functions such as QR decomposition are defined.
// In order to keep the header file a reasonable size, the functions are
// defined in MatrixLinear.h.
//
// 
// .ft B 
//  Note that Matrices run from (1,1) to (n, m), and [0][0] to
//  [n-1][m-1]. The users of the latter form should be careful about sub
//  functions.
//
// ./" The program that this class was orginally used with lots of small
// ./" Matrices.  It was very costly to malloc and free array space every
// ./" time a Matrix is needed.  So, a number of previously freed arrays are
// ./" kept around, and when needed again one of these array is used.  Right
// ./" now, a set of piles of arrays with row <= row_max and col <= col_max
// ./" are kept around.  The piles of kept Matrices can be easily changed.
// ./" Array mallocing and freeing are done only in new_m() and delete_m(),
// ./" so these are the only routines that need to be rewritten.
// 
//  You can do things thinking of a Matrix as a list of numbers.  
//
// .ft B
//  m = hm1.apply(HEP_MATRIX_ELEMENT (*f)(HEP_MATRIX_ELEMENT, int r, int c));
// 
//  applies f to every element of hm1 and places it in m.
//
// .SS See Also:
// SymMatrix[DF].h, GenMatrix[DF].h, DiagMatrix[DF].h Vector[DF].h
// MatrixLinear[DF].h 

#ifndef _Matrix_H_
#define _Matrix_H_

#ifdef GNUPRAGMA
#pragma interface
#endif

#include <vector>

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Matrix/GenMatrix.h"

namespace CLHEP {

class HepRandom;

class HepSymMatrix;
class HepDiagMatrix;
class HepVector;
class HepRotation;

/**
 * @author
 * @ingroup matrix
 */
class HepMatrix : public HepGenMatrix {
public:
   inline HepMatrix();
   // Default constructor. Gives 0 x 0 matrix. Another Matrix can be
   // assigned to it.

   HepMatrix(int p, int q);
   // Constructor. Gives an unitialized p x q matrix.
   HepMatrix(int p, int q, int i);
   // Constructor. Gives an initialized p x q matrix. 
   // If i=0, it is initialized to all 0. If i=1, the diagonal elements
   // are set to 1.0.

   HepMatrix(int p, int q, HepRandom &r);
   // Constructor with a Random object.

   HepMatrix(const HepMatrix &hm1);
   // Copy constructor.

   HepMatrix(const HepSymMatrix &);
   HepMatrix(const HepDiagMatrix &);
   HepMatrix(const HepVector &);
   // Constructors from SymMatrix, DiagMatrix and Vector.

   virtual ~HepMatrix();
   // Destructor.

   virtual int num_row() const;
   // Returns the number of rows.

   virtual int num_col() const;
   // Returns the number of columns.

   virtual const double & operator()(int row, int col) const;
   virtual double & operator()(int row, int col);
   // Read or write a matrix element. 
   // ** Note that the indexing starts from (1,1). **

   HepMatrix & operator *= (double t);
   // Multiply a Matrix by a floating number.

   HepMatrix & operator /= (double t); 
   // Divide a Matrix by a floating number.

   HepMatrix & operator += ( const HepMatrix &);
   HepMatrix & operator += ( const HepSymMatrix &);
   HepMatrix & operator += ( const HepDiagMatrix &);
   HepMatrix & operator += ( const HepVector &);
   HepMatrix & operator -= ( const HepMatrix &);
   HepMatrix & operator -= ( const HepSymMatrix &);
   HepMatrix & operator -= ( const HepDiagMatrix &);
   HepMatrix & operator -= ( const HepVector &);
   // Add or subtract a Matrix. 
   // When adding/subtracting Vector, Matrix must have num_col of one.

   HepMatrix & operator = ( const HepMatrix &);
   HepMatrix & operator = ( const HepSymMatrix &);
   HepMatrix & operator = ( const HepDiagMatrix &);
   HepMatrix & operator = ( const HepVector &);
   HepMatrix & operator = ( const HepRotation &);
   // Assignment operators.

   HepMatrix operator- () const;
   // unary minus, ie. flip the sign of each element.

   HepMatrix apply(double (*f)(double, int, int)) const;
   // Apply a function to all elements of the matrix.

   HepMatrix T() const;
   // Returns the transpose of a Matrix.

   HepMatrix sub(int min_row, int max_row, int min_col, int max_col) const;
   // Returns a sub matrix of a Matrix.
   // WARNING: rows and columns are numbered from 1
   void sub(int row, int col, const HepMatrix &hm1);
   // Sub matrix of this Matrix is replaced with hm1.
   // WARNING: rows and columns are numbered from 1

   friend inline void swap(HepMatrix &hm1, HepMatrix &hm2);
   // Swap hm1 with hm2.

   inline HepMatrix inverse(int& ierr) const;
   // Invert a Matrix. Matrix must be square and is not changed.
   // Returns ierr = 0 (zero) when successful, otherwise non-zero.

   virtual void invert(int& ierr);
   // Invert a Matrix. Matrix must be square.
   // N.B. the contents of the matrix are replaced by the inverse.
   // Returns ierr = 0 (zero) when successful, otherwise non-zero. 
   // This method has less overhead then inverse().

   inline void invert();
   // Invert a matrix. Throw std::runtime_error on failure.

   inline HepMatrix inverse() const;
   // Invert a matrix. Throw std::runtime_error on failure. 


   double determinant() const;
   // calculate the determinant of the matrix.

   double trace() const;
   // calculate the trace of the matrix (sum of diagonal elements).

   class HepMatrix_row {
   public:
      inline HepMatrix_row(HepMatrix&,int);
      double & operator[](int);
   private:
      HepMatrix& _a;
      int _r;
   };
   class HepMatrix_row_const {
   public:
      inline HepMatrix_row_const (const HepMatrix&,int);
      const double & operator[](int) const;
   private:
      const HepMatrix& _a;
      int _r;
   };
   // helper classes for implementing m[i][j]

   inline HepMatrix_row operator[] (int);
   inline const HepMatrix_row_const operator[] (int) const;
   // Read or write a matrix element.
   // While it may not look like it, you simply do m[i][j] to get an
   // element. 
   // ** Note that the indexing starts from [0][0]. **

protected:
   virtual  int num_size() const;
   virtual void invertHaywood4(int& ierr);
   virtual void invertHaywood5(int& ierr);
   virtual void invertHaywood6(int& ierr);

private:
   friend class HepMatrix_row;
   friend class HepMatrix_row_const;
   friend class HepVector;
   friend class HepSymMatrix;
   friend class HepDiagMatrix;
   // Friend classes.

   friend HepMatrix operator+(const HepMatrix &hm1, const HepMatrix &hm2);
   friend HepMatrix operator-(const HepMatrix &hm1, const HepMatrix &hm2);
   friend HepMatrix operator*(const HepMatrix &hm1, const HepMatrix &hm2);
   friend HepMatrix operator*(const HepMatrix &hm1, const HepSymMatrix &hm2);
   friend HepMatrix operator*(const HepMatrix &hm1, const HepDiagMatrix &hm2);
   friend HepMatrix operator*(const HepSymMatrix &hm1, const HepMatrix &hm2);
   friend HepMatrix operator*(const HepDiagMatrix &hm1, const HepMatrix &hm2);
   friend HepMatrix operator*(const HepVector &hm1, const HepMatrix &hm2);
   friend HepVector operator*(const HepMatrix &hm1, const HepVector &hm2);
   friend HepMatrix operator*(const HepSymMatrix &hm1, const HepSymMatrix &hm2);
   // Multiply a Matrix by a Matrix or Vector.

   friend HepVector solve(const HepMatrix &, const HepVector &);
   // solve the system of linear eq
   friend HepVector qr_solve(HepMatrix *, const HepVector &);
   friend HepMatrix qr_solve(HepMatrix *, const HepMatrix &b);
   friend void tridiagonal(HepSymMatrix *a,HepMatrix *hsm);
   friend void row_house(HepMatrix *,const HepMatrix &, double,
			 int, int, int, int);
   friend void row_house(HepMatrix *,const HepVector &, double,
			 int, int);
   friend void back_solve(const HepMatrix &R, HepVector *b);
   friend void back_solve(const HepMatrix &R, HepMatrix *b);
   friend void col_givens(HepMatrix *A, double c,
			  double s, int k1, int k2, 
			  int rowmin, int rowmax);
   //    Does a column Givens update.
   friend void row_givens(HepMatrix *A, double c,
			  double s, int k1, int k2, 
			  int colmin, int colmax);
   friend void col_house(HepMatrix *,const HepMatrix &, double,
			 int, int, int, int);
   friend HepVector house(const HepMatrix &a,int row,int col);
   friend void house_with_update(HepMatrix *a,int row,int col);
   friend void house_with_update(HepMatrix *a,HepMatrix *v,int row,int col);
   friend void house_with_update2(HepSymMatrix *a,HepMatrix *v,
				  int row,int col); 

   int dfact_matrix(double &det, int *ir);
   // factorize the matrix. If successful, the return code is 0. On
   // return, det is the determinant and ir[] is row-interchange
   // matrix. See CERNLIB's DFACT routine.

   int dfinv_matrix(int *ir);
   // invert the matrix. See CERNLIB DFINV.

#ifdef DISABLE_ALLOC
   std::vector<double > m;
#else
   std::vector<double,Alloc<double,25> > m;
#endif
   int nrow, ncol;
   int size_;
};

// Operations other than member functions for Matrix
// implemented in Matrix.cc and Matrix.icc (inline).

HepMatrix operator*(const HepMatrix &, const HepMatrix &);
HepMatrix operator*(double t, const HepMatrix &);
HepMatrix operator*(const HepMatrix &, double );
// Multiplication operators
// Note that m *= hm1 is always faster than m = m * hm1.

HepMatrix operator/(const HepMatrix &, double );
// m = hm1 / t. (m /= t is faster if you can use it.)

HepMatrix operator+(const HepMatrix &hm1, const HepMatrix &hm2);
// m = hm1 + hm2;
// Note that m += hm1 is always faster than m = m + hm1.

HepMatrix operator-(const HepMatrix &hm1, const HepMatrix &hm2);
// m = hm1 - hm2;
// Note that m -= hm1 is always faster than m = m - hm1.

HepMatrix dsum(const HepMatrix&, const HepMatrix&);
// Direct sum of two matrices. The direct sum of A and B is the matrix 
//        A 0
//        0 B

HepVector solve(const HepMatrix &, const HepVector &);
// solve the system of linear equations using LU decomposition.

std::ostream& operator<<(std::ostream &s, const HepMatrix &q);
// Read in, write out Matrix into a stream.
 
//
// Specialized linear algebra functions
//

HepVector qr_solve(const HepMatrix &A, const HepVector &b);
HepVector qr_solve(HepMatrix *A, const HepVector &b);
HepMatrix qr_solve(const HepMatrix &A, const HepMatrix &b);
HepMatrix qr_solve(HepMatrix *A, const HepMatrix &b);
// Works like backsolve, except matrix does not need to be upper
// triangular. For nonsquare matrix, it solves in the least square sense.

HepMatrix qr_inverse(const HepMatrix &A);
HepMatrix qr_inverse(HepMatrix *A);
// Finds the inverse of a matrix using QR decomposition.  Note, often what
// you really want is solve or backsolve, they can be much quicker than
// inverse in many calculations.


void qr_decomp(HepMatrix *A, HepMatrix *hsm);
HepMatrix qr_decomp(HepMatrix *A);
// Does a QR decomposition of a matrix.

void back_solve(const HepMatrix &R, HepVector *b);
void back_solve(const HepMatrix &R, HepMatrix *b);
// Solves R*x = b where R is upper triangular.  Also has a variation that
// solves a number of equations of this form in one step, where b is a matrix
// with each column a different vector. See also solve.

void col_house(HepMatrix *a, const HepMatrix &v, double vnormsq,
	       int row, int col, int row_start, int col_start);
void col_house(HepMatrix *a, const HepMatrix &v, int row, int col,
	       int row_start, int col_start);
// Does a column Householder update.

void col_givens(HepMatrix *A, double c, double s,
		int k1, int k2, int row_min=1, int row_max=0);
// do a column Givens update

void row_givens(HepMatrix *A, double c, double s,
		int k1, int k2, int col_min=1, int col_max=0);
// do a row Givens update

void givens(double a, double b, double *c, double *s);
// algorithm 5.1.5 in Golub and Van Loan

HepVector house(const HepMatrix &a, int row=1, int col=1);
// Returns a Householder vector to zero elements.

void house_with_update(HepMatrix *a, int row=1, int col=1);
void house_with_update(HepMatrix *a, HepMatrix *v, int row=1, int col=1);
// Finds and does Householder reflection on matrix.

void row_house(HepMatrix *a, const HepVector &v, double vnormsq,
	       int row=1, int col=1);
void row_house(HepMatrix *a, const HepMatrix &v, double vnormsq,
	       int row, int col, int row_start, int col_start);
void row_house(HepMatrix *a, const HepMatrix &v, int row, int col,
	       int row_start, int col_start);
// Does a row Householder update.

}  // namespace CLHEP

#ifdef ENABLE_BACKWARDS_COMPATIBILITY
//  backwards compatibility will be enabled ONLY in CLHEP 1.9
using namespace CLHEP;
#endif

#ifndef HEP_DEBUG_INLINE
#include "CLHEP/Matrix/Matrix.icc"
#endif

#endif /*_Matrix_H*/
