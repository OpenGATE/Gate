/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATECOMPTONPB_HH
#define GATECOMPTONPB_HH


#include "GateVProcess.hh"

#include "G4ComptonScattering.hh"


#ifdef G4VERSION9_3
MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GateComptonPB)
#else
MAKE_PROCESS_AUTO_CREATOR(GateComptonPB)
#endif

#endif
