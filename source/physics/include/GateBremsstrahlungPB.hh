/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEBREMPB_HH
#define GATEBREMPB_HH


#include "GateVProcess.hh"

#include "G4eBremsstrahlung.hh"

#ifdef G4VERSION9_3
MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GateBremsstrahlungPB)
#else
MAKE_PROCESS_AUTO_CREATOR(GateBremsstrahlungPB)
#endif


#endif
