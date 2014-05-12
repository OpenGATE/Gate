/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEFICTITIOUSPROCESSPB_HH
#define GATEFICTITIOUSPROCESSPB_HH 1


#include "GateVProcess.hh"

#include "G4LivermoreRayleighModel.hh"
#include "G4LivermorePhotoElectricModel.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermoreComptonModel.hh"
#include "G4ComptonScattering.hh"
#include "G4LivermoreGammaConversionModel.hh"
#include "G4GammaConversion.hh"

#include "GateTotalDiscreteProcess.hh"
#include "GateFictitiousFastSimulationModel.hh"
#include "GatePETVRTSettings.hh"
#include "GatePETVRTManager.hh"
#include "G4FastSimulationManagerProcess.hh"


MAKE_PROCESS_AUTO_CREATOR(GateFictitiousProcessPB)

#endif
