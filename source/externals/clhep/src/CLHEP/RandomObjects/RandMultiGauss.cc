// $Id: RandMultiGauss.cc,v 1.3 2003/08/13 20:00:13 garren Exp $
// -----------------------------------------------------------------------
//                             HEP Random
//                          --- RandMultiGauss ---
//                      class implementation file
// -----------------------------------------------------------------------

// =======================================================================
// Mark Fischler  - Created: 17th September 1998
// =======================================================================

// Some theory about how to get the Multivariate Gaussian from a bunch 
// of Gaussian deviates.  For the purpose of this discussion, take mu = 0.
//
// We want an n-vector x with distribution (See table 28.1 of Review of PP)
//
//            exp ( .5 * x.T() * S.inverse() * x ) 
//   f(x;S) = ------------------------------------
//                 sqrt ( (2*pi)^n * S.det() )
//
// Suppose S = U * D * U.T() with U orthogonal ( U*U.T() = 1 ) and D diagonal.
// Consider a random n-vector y such that each element y(i)is distributed as a
// Gaussian with sigma = sqrt(D(i,i)).  Then the distribution of y is the
// product of n Gaussains which can be written as 
//
//            exp ( .5 * y.T() * D.inverse() * y ) 
//   f(y;D) = ------------------------------------
//                 sqrt ( (2*pi)^n * D.det() )
//
// Now take an n-vector x = U * y (or y = U.T() * x ).  Then substituting,
//
//            exp ( .5 * x * U * D.inverse() U.T() * x ) 
// f(x;D,U) = ------------------------------------------
//                 sqrt ( (2*pi)^n * D.det() )
//
// and this simplifies to the desired f(x;S) when we note that 
//	D.det() = S.det()  and U * D.inverse() * U.T() = S.inverse()
//
// So the strategy is to diagonalize S (finding U and D), form y with each 
// element a Gaussian random with sigma of sqrt(D(i,i)), and form x = U*y.
// (It turns out that the D information needed is the sigmas.)
// Since for moderate or large n the work needed to diagonalize S can be much
// greater than the work to generate n Gaussian variates, we save the U and
// sigmas for both the default S and the latest S value provided.  

#include "CLHEP/RandomObjects/RandMultiGauss.h"
#include "CLHEP/RandomObjects/defs.h"
#include <cmath>	// for log()

namespace CLHEP {

// ------------
// Constructors
// ------------

RandMultiGauss::RandMultiGauss( HepRandomEngine & anEngine, 
				const HepVector & mu,
				const HepSymMatrix & S ) 
      : localEngine(&anEngine), 
	deleteEngine(false), 
	set(false),
	nextGaussian(0.0)
{ 
  if (S.num_row() != mu.num_row()) {
    std::cerr << "In constructor of RandMultiGauss distribution: \n" <<
            "      Dimension of mu (" << mu.num_row() << 
	    ") does not match dimension of S (" << S.num_row() << ")\n";
    std::cerr << "---Exiting to System\n";
    exit(1);
  }
  defaultMu = mu;
  defaultSigmas = HepVector(S.num_row());
  prepareUsigmas (S,  defaultU, defaultSigmas);
}
  
RandMultiGauss::RandMultiGauss( HepRandomEngine * anEngine, 
				const HepVector & mu,
				const HepSymMatrix & S ) 
      : localEngine(anEngine), 
	deleteEngine(true), 
	set(false),
	nextGaussian(0.0)
{ 
  if (S.num_row() != mu.num_row()) {
    std::cerr << "In constructor of RandMultiGauss distribution: \n" <<
            "      Dimension of mu (" << mu.num_row() << 
	    ") does not match dimension of S (" << S.num_row() << ")\n";
    std::cerr << "---Exiting to System\n";
    exit(1);
  }
  defaultMu = mu;
  defaultSigmas = HepVector(S.num_row());
  prepareUsigmas (S,  defaultU, defaultSigmas);
}
  
RandMultiGauss::RandMultiGauss( HepRandomEngine & anEngine ) 
      : localEngine(&anEngine), 
	deleteEngine(false), 
	set(false),
	nextGaussian(0.0)
{ 
  defaultMu = HepVector(2,0);
  defaultU  = HepMatrix(2,1);
  defaultSigmas = HepVector(2);
  defaultSigmas(1) = 1.;
  defaultSigmas(2) = 1.;
}

RandMultiGauss::RandMultiGauss( HepRandomEngine * anEngine ) 
      : localEngine(anEngine), 
	deleteEngine(true), 
	set(false),
	nextGaussian(0.0)
{ 
  defaultMu = HepVector(2,0);
  defaultU  = HepMatrix(2,1);
  defaultSigmas = HepVector(2);
  defaultSigmas(1) = 1.;
  defaultSigmas(2) = 1.;
}

RandMultiGauss::~RandMultiGauss() {
  if ( deleteEngine ) delete localEngine;
}

// ----------------------------
// prepareUsigmas()
// ----------------------------

void RandMultiGauss::prepareUsigmas( const HepSymMatrix & S,
                   	    	HepMatrix & U, 
	                    	HepVector & sigmas ) { 
 
  HepSymMatrix tempS ( S ); // Since diagonalize does not take a const s
			    // we have to copy S.

  U = diagonalize ( &tempS );  			// S = U Sdiag U.T()
  HepSymMatrix D = S.similarityT(U);		// D = U.T() S U = Sdiag
  for (int i = 1; i <= S.num_row(); i++) {
    double s2 = D(i,i);
    if ( s2 > 0 ) {
	sigmas(i) = sqrt ( s2 );
    } else {
      std::cerr << "In RandMultiGauss distribution: \n" <<
            "      Matrix S is not positive definite.  Eigenvalues are:\n";
      for (int ixx = 1; ixx <= S.num_row(); ixx++) {
	std::cerr << "      " << D(ixx,ixx) << std::endl;
      }
      std::cerr << "---Exiting to System\n";
      exit(1);
    }
  }
} // prepareUsigmas

// -----------
// deviates()
// -----------

HepVector RandMultiGauss::deviates ( 	const HepMatrix & U,
					const HepVector & sigmas,
					HepRandomEngine * engine,
		                        bool& available,
					double& next)
{
  // Returns vector of gaussian randoms based on sigmas, rotated by U,
  // with means of 0.

  int n = sigmas.num_row(); 
  HepVector v(n);  // The vector to be returned

  double r,v1,v2,fac;
  
  int i = 1;
  if (available) {
    v(1) = next;
    i = 2;
    available = false;
  }
    
  while ( i <= n ) {
    do {
      v1 = 2.0 * engine->flat() - 1.0;
      v2 = 2.0 * engine->flat() - 1.0;
      r = v1*v1 + v2*v2;
    } while ( r > 1.0 );
    fac = sqrt(-2.0*log(r)/r);
    v(i++) = v1*fac;
    if ( i <= n ) {
      v(i++) = v2*fac;
    } else {
      next = v2*fac;
      available = true;
    }
  } 

  for ( i = 1; i <= n; i++ ) {
    v(i) *= sigmas(i);
  }

  return U*v;

} // deviates() 

// ---------------
// fire signatures
// ---------------

HepVector RandMultiGauss::fire() {
  // Returns a pair of unit normals, using the S and mu set in constructor,
  // utilizing the engine belonging to this instance of RandMultiGauss.

  return defaultMu + deviates ( defaultU, defaultSigmas, 
			        localEngine, set, nextGaussian );

} // fire();


HepVector RandMultiGauss::fire( const HepVector& mu, const HepSymMatrix& S ) {

  HepMatrix U;  
  HepVector sigmas;

  if (mu.num_row() == S.num_row()) {
    prepareUsigmas ( S, U, sigmas );
    return mu + deviates ( U, sigmas, localEngine, set, nextGaussian );
  } else {
    std::cerr << "In firing RandMultiGauss distribution with explicit mu and S: \n"
         << "      Dimension of mu (" << mu.num_row() << 
	    ") does not match dimension of S (" << S.num_row() << ")\n";
    std::cerr << "---Exiting to System\n";
    exit(1);
  }
  return mu;    // This line cannot be reached.  But without returning 
		// some HepVector here, KCC 3.3 complains.

} // fire(mu, S);


// --------------------
// fireArray signatures
// --------------------

void RandMultiGauss::fireArray( const int size, HepVector* array ) {

  int i;
  for (i = 0; i < size; ++i) {
    array[i] = defaultMu + deviates ( defaultU, defaultSigmas, 
			        localEngine, set, nextGaussian );
  } 

} // fireArray ( size, vect )


void RandMultiGauss::fireArray( const int size, HepVector* array,
                                 const HepVector& mu, const HepSymMatrix& S ) {

  // For efficiency, we diagonalize S once and generate all the vectors based
  // on that U and sigmas.

  HepMatrix U;  
  HepVector sigmas;
  HepVector mu_ (mu);

  if (mu.num_row() == S.num_row()) {
    prepareUsigmas ( S, U, sigmas );
  } else {
    std::cerr << 
    "In fireArray for RandMultiGauss distribution with explicit mu and S: \n"
         << "      Dimension of mu (" << mu.num_row() << 
	    ") does not match dimension of S (" << S.num_row() << ")\n";
    std::cerr << "---Exiting to System\n";
    exit(1);
  }

  int i;
  for (i=0; i<size; ++i) {
    array[i] = mu_ + deviates(U, sigmas, localEngine, set, nextGaussian);
  }

} // fireArray ( size, vect, mu, S )

// ----------
// operator()
// ----------

HepVector RandMultiGauss::operator()() {
  return fire();
} 

HepVector RandMultiGauss::operator()
			( const HepVector& mu, const HepSymMatrix& S ) {
  return fire(mu,S);
} 


}  // namespace CLHEP
