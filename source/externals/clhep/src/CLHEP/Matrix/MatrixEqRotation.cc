// -*- C++ -*-
// ---------------------------------------------------------------------------
//
// This file is a part of the CLHEP - a Class Library for High Energy Physics.
//

#include "CLHEP/Matrix/defs.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Vector/Rotation.h"

#include <iostream>

namespace CLHEP {

HepMatrix & HepMatrix::operator=(const HepRotation &hm1) {
  if(9!=size_) {
    //delete &m;
    size_ = 9;
    m.resize(size_);
  }
  nrow = ncol = 3;
  mIter hmm1;
  hmm1 = m.begin();
  *hmm1++ = hm1.xx();
  *hmm1++ = hm1.xy();
  *hmm1++ = hm1.xz();
  *hmm1++ = hm1.yx();
  *hmm1++ = hm1.yy();
  *hmm1++ = hm1.yz();
  *hmm1++ = hm1.zx();
  *hmm1++ = hm1.zy();
  *hmm1   = hm1.zz();
  return (*this);
}

}  // namespace CLHEP
