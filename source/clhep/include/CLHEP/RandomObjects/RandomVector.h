// -*- C++ -*-
//
// -----------------------------------------------------------------------
//                             HEP Random
//                       --- HepRandomVector ---
//                          class header file
// -----------------------------------------------------------------------
// This file is part of CLHEP, extended to match the distributions in RPP.
//
// It's exactly analogous to HepRandom except that the return types for
// the fire() and related methods are std::vector<double> instead of
// double.
//
// Distribution classes returning HepVectors of results inherit from
// HepRandomVector instead of HepRandom.
//
//	HepVector is used instead of the more modern looking
//	std::vector<double> because the motivating sub-class
//	RandMultiGauss uses HepMatrix to supply the correlation
//	matrix S anyway.  Given that, we might as well stick to
//	HepVector when a vector of numbers is needed, as well.
//
// =======================================================================
// Mark Fischler  - Created: 19 Oct, 1998
//    10/20/98	  - Removed all shoot-related material
// =======================================================================

#ifndef HepRandomVector_h
#define HepRandomVector_h 1

#include "CLHEP/RandomObjects/defs.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Matrix/Vector.h"

namespace CLHEP {

/**
 * @author Mark Fischler <mf@fnal.gov>
 * @ingroup robjects
 */
class HepRandomVector {

public:

  HepRandomVector();
  HepRandomVector(long seed);
  // Contructors with and without a seed using a default engine
  // (JamesRandom) which is instantiated for use of this distribution
  // instance.  If the seed is omitted, multiple instantiations will
  // each get unique seeds.

  HepRandomVector(HepRandomEngine & engine);
  HepRandomVector(HepRandomEngine * engine);
  // Constructor taking an alternative engine as argument. If a pointer is
  // given the corresponding object will be deleted by the HepRandom
  // destructor.

  virtual ~HepRandomVector();
  // Destructor

  inline HepVector flat();
  // Returns vector of flat values ( interval ]0.1[ ).

  inline HepVector flat (HepRandomEngine* theNewEngine);
  // Returns a vector of flat values, given a defined Random Engine.

  inline void flatArray(const int size, HepVector* vect);
  // Fills "vect" array of flat random values, given the size.
  // Included for consistency with the HepRandom class.

  inline void flatArray(HepRandomEngine* theNewEngine,
                        const int size, HepVector* vect);
  // Fills "vect" array of flat random values, given the size
  // and a defined Random Engine.


  virtual HepVector operator()();
  // To get a flat random number using the operator ().


private:       // -------- Private methods ---------

  inline void setSeed(long seed, int lux);
  // (Re)Initializes the generator with a seed.

  inline long getSeed() const;
  // Gets the current seed of the current generator.

  inline void setSeeds(const long* seeds, int aux);
  // (Re)Initializes the generator with a zero terminated list of seeds.

  inline const long* getSeeds () const;
  // Gets the current array of seeds of the current generator.

  void setEngine (HepRandomEngine* engine) { theEngine = engine; }
  // To set the underlying algorithm object

  HepRandomEngine * getEngine() const { return theEngine; }
  // Returns a pointer to the underlying algorithm object.

  void saveStatus( const char filename[] = "Config.conf" ) const;
  // Saves to file the current status of the current engine.

  void restoreStatus( const char filename[] = "Config.conf" );
  // Restores a saved status (if any) for the current engine.

  void showStatus() const;
  // Dumps the current engine status on screen.

protected:     // -------- Data members ---------

  HepRandomEngine * theEngine;
  // The corresponding algorithm.

private:       // -------- Data members ---------

  bool deleteEngine;
  // True if the engine should be deleted on destruction.

};

}  // namespace CLHEP

#include "CLHEP/RandomObjects/RandomVector.icc"

#ifdef ENABLE_BACKWARDS_COMPATIBILITY
//  backwards compatibility will be enabled ONLY in CLHEP 1.9
using namespace CLHEP;
#endif

#endif
