// $Id: RandMultiGauss.h,v 1.3 2003/10/23 21:29:51 garren Exp $
// -*- C++ -*-
//
// -----------------------------------------------------------------------
//                             HEP Random
//                          --- RandMultiGauss ---
//                          class header file
// -----------------------------------------------------------------------

// Class defining methods for firing multivariate gaussian distributed
// random values, given a vector of means and a covariance matrix
// Definitions are those from 1998 Review of Particle Physics, section 28.3.3.
//
// This utilizes the following other comonents of CLHEP:
//	RandGauss from the Random package to get individual deviates
//	HepVector, HepSymMatrix and HepMatrix from the Matrix package
//	HepSymMatrix::similarity(HepMatrix)
//	diagonalize(HepSymMatrix *s)
// The author of this distribution relies on diagonalize() being correct.
//
// Although original distribution classes in the Random package return a
// double when fire() (or operator()) is done, RandMultiGauss returns a
// HepVector of values.
//
// =======================================================================
// Mark Fischler  - Created: 19th September 1999
// =======================================================================

#ifndef RandMultiGauss_h
#define RandMultiGauss_h 1

#include "CLHEP/RandomObjects/defs.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/RandomObjects/RandomVector.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"

namespace CLHEP {

/**
 * @author Mark Fischler <mf@fnal.gov>
 * @ingroup robjects
 */
class RandMultiGauss : public HepRandomVector {

public:

  RandMultiGauss ( HepRandomEngine& anEngine,
		   const HepVector& mu,
                   const HepSymMatrix& S );
		// The symmetric matrix S MUST BE POSITIVE DEFINITE
		// and MUST MATCH THE SIZE OF MU.

  RandMultiGauss ( HepRandomEngine* anEngine,
		   const HepVector& mu,
                   const HepSymMatrix& S );
		// The symmetric matrix S MUST BE POSITIVE DEFINITE
		// and MUST MATCH THE SIZE OF MU.

  // These constructors should be used to instantiate a RandMultiGauss
  // distribution object defining a local engine for it.
  // The static generator will be skipped using the non-static methods
  // defined below.
  // If the engine is passed by pointer the corresponding engine object
  // will be deleted by the RandMultiGauss destructor.
  // If the engine is passed by reference the corresponding engine object
  // will not be deleted by the RandGauss destructor.

  // The following are provided for convenience in the case where each
  // random fired will have a different mu and S.  They set the default mu
  // to the zero 2-vector, and the default S to the Unit 2x2 matrix.
  RandMultiGauss ( HepRandomEngine& anEngine );
  RandMultiGauss ( HepRandomEngine* anEngine );

  virtual ~RandMultiGauss();
  // Destructor

  HepVector fire();

  HepVector fire( const HepVector& mu, const HepSymMatrix& S );
		// The symmetric matrix S MUST BE POSITIVE DEFINITE
		// and MUST MATCH THE SIZE OF MU.

  // A note on efficient usage when firing many vectors of Multivariate
  // Gaussians:   For n > 2 the work needed to diagonalize S is significant.
  // So if you only want a collection of uncorrelated Gaussians, it will be
  // quicker to generate them one at a time.
  //
  // The class saves the diagonalizing matrix for the default S.
  // Thus generating vectors with that same S can be quite efficient.
  // If you require a small number of different S's, known in
  // advance, consider instantiating RandMulitGauss for each different S,
  // sharing the same engine.
  //
  // If you require a random using the default S for a distribution but a
  // different mu, it is most efficient to imply use the default fire() and
  // add the difference of the mu's to the returned HepVector.

  void fireArray ( const int size, HepVector* array);
  void fireArray ( const int size, HepVector* array,
		   const HepVector& mu, const HepSymMatrix& S );

  HepVector operator()();
  HepVector operator()( const HepVector& mu, const HepSymMatrix& S );
		// The symmetric matrix S MUST BE POSITIVE DEFINITE
		// and MUST MATCH THE SIZE OF MU.

private:

  // Private copy constructor. Defining it here disallows use.
  RandMultiGauss(const RandMultiGauss &d);

  HepRandomEngine* localEngine;
  bool deleteEngine;
  HepVector    defaultMu;
  HepMatrix    defaultU;
  HepVector    defaultSigmas;	// Each sigma is sqrt(D[i,i])

  bool set;
  double nextGaussian;

  static void prepareUsigmas (  const HepSymMatrix & S,
		   		HepMatrix & U,
		   		HepVector & sigmas );

  static HepVector deviates ( const HepMatrix & U,
		       	      const HepVector & sigmas,
		       	      HepRandomEngine * engine,
		       	      bool& available,
		 	      double& next);
  // Returns vector of gaussian randoms based on sigmas, rotated by U,
  // with means of 0.

};

}  // namespace CLHEP

#ifdef ENABLE_BACKWARDS_COMPATIBILITY
//  backwards compatibility will be enabled ONLY in CLHEP 1.9
using namespace CLHEP;
#endif

#endif // RandMultiGauss_h
