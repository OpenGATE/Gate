/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifndef GATEGAMMACONVERSIONPB_HH
#define GATEGAMMACONVERSIONPB_HH


#include "GateVProcess.hh"

#include "G4GammaConversion.hh"
#include "G4LivermoreGammaConversionModel.hh"
#include "G4LivermorePolarizedGammaConversionModel.hh"
#include "G4PenelopeGammaConversionModel.hh"

MAKE_PROCESS_AUTO_CREATOR_WITH_MODEL(GateGammaConversionPB)

#endif
