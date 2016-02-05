/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifndef GATEPHOTOELECTRICPB_HH
#define GATEPHOTOELECTRICPB_HH

#include "GateVProcess.hh"

#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"
#include "G4PenelopePhotoElectricModel.hh"
#include "G4LivermorePolarizedPhotoElectricModel.hh"

MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GatePhotoElectricPB)

#endif
