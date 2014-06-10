// -*- C++ -*-
// CLASSDOC OFF
// ---------------------------------------------------------------------------
// CLASSDOC ON
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//
//   Although Vector and Matrix class are very much related, I like the typing
//   information I get by making them different types. It is usually an error
//   to use a Matrix where a Vector is expected, except in the case that the
//   Matrix is a single column.  But this case can be taken care of by using
//   constructors as conversions.  For this same reason, I don't want to make
//   a Vector a derived class of Matrix.
//

#ifndef _Vector_H_
#define _Vector_H_

#ifdef GNUPRAGMA
#pragma interface
#endif

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Matrix/GenMatrix.h"

namespace CLHEP {

class HepRandom;

class HepMatrix;
class HepSymMatrix;
class HepDiagMatrix;
class Hep3Vector;

/**
 * @author
 * @ingroup matrix
 */
class HepVector : public HepGenMatrix {
public:
   inline HepVector();
   // Default constructor. Gives vector of length 0.
   // Another Vector can be assigned to it.

   explicit HepVector(int p);
   HepVector(int p, int);
   // Constructor. Gives vector of length p.

   HepVector(int p, HepRandom &r);

   HepVector(const HepVector &v);
   HepVector(const HepMatrix &m);
   // Copy constructors.
   // Note that there is an assignment operator for v = Hep3Vector.

   virtual ~HepVector();
   // Destructor.

   inline const double & operator()(int row) const;
   inline double & operator()(int row);
   // Read or write a matrix element.
   // ** Note that the indexing starts from (1). **

   inline const double & operator[](int row) const;
   inline double & operator[](int row);
   // Read and write an element of a Vector.
   // ** Note that the indexing starts from [0]. **

   virtual const double & operator()(int row, int col) const;
   virtual double & operator()(int row, int col);
   // Read or write a matrix element.
   // ** Note that the indexing starts from (1,1). **
   // Allows accessing Vector using GenMatrix

   HepVector & operator*=(double t);
   // Multiply a Vector by a floating number.

   HepVector & operator/=(double t);
   // Divide a Vector by a floating number.

   HepVector & operator+=( const HepMatrix &v2);
   HepVector & operator+=( const HepVector &v2);
   HepVector & operator-=( const HepMatrix &v2);
   HepVector & operator-=( const HepVector &v2);
   // Add or subtract a Vector.

   HepVector & operator=( const HepVector &hm2);
   // Assignment operators.

   HepVector& operator=(const HepMatrix &);
   HepVector& operator=(const Hep3Vector &);
   // assignment operators from other classes.

   HepVector operator- () const;
   // unary minus, ie. flip the sign of each element.

   HepVector apply(double (*f)(double, int)) const;
   // Apply a function to all elements.

   HepVector sub(int min_row, int max_row) const;
   // Returns a sub vector.
   HepVector sub(int min_row, int max_row);
   // SGI CC bug. I have to have both with/without const. I should not need
   // one without const.

   void sub(int row, const HepVector &v1);
   // Replaces a sub vector of a Vector with v1.

   inline double normsq() const;
   // Returns norm squared.

   inline double norm() const;
   // Returns norm.

   virtual int num_row() const;
   // Returns number of rows.

   virtual int num_col() const;
   // Number of columns. Always returns 1. Provided for compatibility with
   // GenMatrix.

   HepMatrix T() const;
   // Returns the transpose of a Vector. Note that the returning type is
   // Matrix.

   friend inline void swap(HepVector &v1, HepVector &v2);
   // Swaps two vectors.

protected:
   virtual int num_size() const;

private:
   virtual void invert(int&);
   // produces an error. Demanded by GenMatrix

   friend class HepDiagMatrix;
   friend class HepSymMatrix;
   friend class HepMatrix;
   // friend classes

   friend double dot(const HepVector &v1, const HepVector &v2);
   // f = v1 * v2;

   friend HepVector operator+(const HepVector &v1, const HepVector &v2);
   friend HepVector operator-(const HepVector &v1, const HepVector &v2);
   friend HepVector operator*(const HepSymMatrix &hm1, const HepVector &hm2);
   friend HepVector operator*(const HepDiagMatrix &hm1, const HepVector &hm2);
   friend HepMatrix operator*(const HepVector &hm1, const HepMatrix &hm2);
   friend HepVector operator*(const HepMatrix &hm1, const HepVector &hm2);

   friend HepVector solve(const HepMatrix &a, const HepVector &v);
   friend void tridiagonal(HepSymMatrix *a,HepMatrix *hsm);
   friend void row_house(HepMatrix *,const HepMatrix &, double, int, int,
			 int, int);
   friend void row_house(HepMatrix *,const HepVector &, double, int, int);
   friend void back_solve(const HepMatrix &R, HepVector *b);
   friend void col_house(HepMatrix *,const HepMatrix &,double, int, int,
			 int, int);
   friend HepVector house(const HepSymMatrix &a,int row,int col);
   friend HepVector house(const HepMatrix &a,int row,int col);
   friend void house_with_update(HepMatrix *a,int row,int col);
   friend HepSymMatrix vT_times_v(const HepVector &v);
   friend HepVector qr_solve(HepMatrix *, const HepVector &);

#ifdef DISABLE_ALLOC
   std::vector<double > m;
#else
   std::vector<double,Alloc<double,25> > m;
#endif
   int nrow;
};

//
// Operations other than member functions
//

std::ostream& operator<<(std::ostream &s, const HepVector &v);
// Write out Matrix, SymMatrix, DiagMatrix and Vector into ostream.

HepVector operator*(const HepMatrix &hm1, const HepVector &hm2);
HepVector operator*(double t, const HepVector &v1);
HepVector operator*(const HepVector &v1, double t);
// Multiplication operators.
// Note that m *= x is always faster than m = m * x.

HepVector operator/(const HepVector &v1, double t);
// Divide by a real number.

HepVector operator+(const HepMatrix &hm1, const HepVector &v2);
HepVector operator+(const HepVector &v1, const HepMatrix &hm2);
HepVector operator+(const HepVector &v1, const HepVector &v2);
// Addition operators

HepVector operator-(const HepMatrix &hm1, const HepVector &v2);
HepVector operator-(const HepVector &v1, const HepMatrix &hm2);
HepVector operator-(const HepVector &v1, const HepVector &v2);
// subtraction operators

HepVector dsum(const HepVector &s1, const HepVector &s2);
// Direct sum of two vectors;

}  // namespace CLHEP

#ifdef ENABLE_BACKWARDS_COMPATIBILITY
//  backwards compatibility will be enabled ONLY in CLHEP 1.9
using namespace CLHEP;
#endif

#include "CLHEP/Matrix/Vector.icc"

#endif /*!_Vector_H*/
