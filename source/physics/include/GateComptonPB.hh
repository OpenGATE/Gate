/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATECOMPTONPB_HH
#define GATECOMPTONPB_HH

#include "GateConfiguration.h"
#include "GateVProcess.hh"

#include "G4ComptonScattering.hh"
#include "G4LivermoreComptonModel.hh"
#include "G4LivermorePolarizedComptonModel.hh"
#include "G4PenelopeComptonModel.hh"


MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GateComptonPB)

#endif
