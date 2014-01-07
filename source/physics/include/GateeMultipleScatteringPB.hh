/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifndef GATEEMULTISCATTERINGPB_HH
#define GATEEMULTISCATTERINGPB_HH

#include "GateVProcess.hh"

#include "G4eMultipleScattering.hh"
#include "G4UrbanMscModel.hh" //replace model93 and model95 in geant4.10
//#include "G4UrbanMscModel93.hh"
//#include "G4UrbanMscModel95.hh"

MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GateeMultipleScatteringPB)

#endif
