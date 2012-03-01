/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEPHOTOELECTRICPB_HH
#define GATEPHOTOELECTRICPB_HH


#include "GateVProcess.hh"

#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"
#include "G4PenelopePhotoElectricModel.hh"
#include "G4LivermorePolarizedPhotoElectricModel.hh"

#ifdef G4VERSION9_3
MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GatePhotoElectricPB)
#else
MAKE_PROCESS_AUTO_CREATOR(GatePhotoElectricPB)
#endif

#endif
