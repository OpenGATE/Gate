/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateConfiguration.h"
#ifndef GATEEIONISATIONPB_HH
#define GATEEIONISATIONPB_HH


#include "GateVProcess.hh"

#include "G4eIonisation.hh"
#include "G4LivermoreIonisationModel.hh"
#include "G4PenelopeIonisationModel.hh"

#ifdef G4VERSION9_3
MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GateElectronIonisationPB)
#else
MAKE_PROCESS_AUTO_CREATOR(GateElectronIonisationPB)
#endif

#endif
