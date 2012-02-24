/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEPOSITRONANNIHILATIONSTDPB_HH
#define GATEPOSITRONANNIHILATIONSTDPB_HH


#include "GateVProcess.hh"

#include "G4eplusAnnihilation.hh"


#ifdef G4VERSION9_3
MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GatePositronAnnihilationStdPB)
#else
MAKE_PROCESS_AUTO_CREATOR(GatePositronAnnihilationStdPB)
#endif

#endif
