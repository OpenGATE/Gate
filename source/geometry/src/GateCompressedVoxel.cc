/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateCompressedVoxel.hh"

// Output operator (prototype and implementation)

std::ostream& operator << (std::ostream& os, const GateCompressedVoxel& v) {
  os << ' '<<v[0]<<' '<<v[1]<<' '<<v[2]<<' '<<v[3]<<' '<<v[4]<<' '<<v[5]<<' '<<v[6]<<' ';
  return os;
}



