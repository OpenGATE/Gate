// $Id: RandomVector.cc,v 1.3 2003/08/13 20:00:13 garren Exp $
// -----------------------------------------------------------------------
//                             HEP Random
//                        --- HepRandomVector ---
//                      class implementation file
// -----------------------------------------------------------------------
// =======================================================================
// Mark Fischler  - Created: 19 OCtober, 1998
// =======================================================================

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/RandomObjects/RandomVector.h"
#include "CLHEP/RandomObjects/defs.h"

namespace CLHEP {

//------------------------- HepRandomVector ---------------------------------

HepRandomVector::HepRandomVector()
: theEngine(new HepJamesRandom(11327503L)), deleteEngine(true)
{
}

HepRandomVector::HepRandomVector(long seed)
: theEngine(new HepJamesRandom(seed)), deleteEngine(true) {
}

HepRandomVector::HepRandomVector(HepRandomEngine & engine)
: theEngine(&engine), deleteEngine(false) {
}

HepRandomVector::HepRandomVector(HepRandomEngine * engine)
: theEngine(engine), deleteEngine(true) {
}

HepRandomVector::~HepRandomVector() {
  if ( deleteEngine ) delete theEngine;
}

HepVector HepRandomVector::operator()() {
  return flat();
}

}  // namespace CLHEP

