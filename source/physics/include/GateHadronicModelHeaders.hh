/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

// Elastic
#if (G4VERSION_MAJOR == 9)
#include "G4LElastic.hh" 
#endif

#include "G4ElasticHadrNucleusHE.hh"
#include "G4LEpp.hh"
#include "G4LEnp.hh"
#include "G4HadronElastic.hh"

// Nucleus-nucleus
#include "G4BinaryLightIonReaction.hh"

#if (G4VERSION_MAJOR == 9)
#include "G4LEDeuteronInelastic.hh"
#include "G4LETritonInelastic.hh"
#include "G4LEAlphaInelastic.hh"
#endif

#include "G4WilsonAbrasionModel.hh"
#include "G4EMDissociation.hh"
#include "G4QMDReaction.hh"

// Low energy parameterized
#if (G4VERSION_MAJOR == 9)
#include "G4LEProtonInelastic.hh"
#include "G4LEPionPlusInelastic.hh"
#include "G4LEPionMinusInelastic.hh"
#include "G4LEKaonPlusInelastic.hh"
#include "G4LEKaonMinusInelastic.hh"
#include "G4LEKaonZeroLInelastic.hh"
#include "G4LEKaonZeroSInelastic.hh"
#include "G4LENeutronInelastic.hh"
#include "G4LELambdaInelastic.hh"
#include "G4LESigmaPlusInelastic.hh"
#include "G4LESigmaMinusInelastic.hh"
#include "G4LEXiMinusInelastic.hh"
#include "G4LEXiZeroInelastic.hh"
#include "G4LEOmegaMinusInelastic.hh"
#include "G4LEAntiProtonInelastic.hh"
#include "G4LEAntiNeutronInelastic.hh"
#include "G4LEAntiLambdaInelastic.hh"
#include "G4LEAntiSigmaPlusInelastic.hh"
#include "G4LEAntiSigmaMinusInelastic.hh"
#include "G4LEAntiXiMinusInelastic.hh"
#include "G4LEAntiXiZeroInelastic.hh"
#include "G4LEAntiOmegaMinusInelastic.hh"
#endif

// Leading Particle Bias
// Seb Modifs - 05/11/2010 - Geant4 9.4 validations
//#include "G4Mars5GeV.hh"

// Precompound
#include "G4ExcitationHandler.hh"
#include "G4PreCompoundModel.hh"

// Cascade
#include "G4CascadeInterface.hh"
#include "G4BinaryCascade.hh"
#include "GateBinaryCascade.hh"

// Gamma- and Lepto-Nuclear
#if (G4VERSION_MAJOR == 9)
#include "G4ElectroNuclearReaction.hh"
#include "G4GammaNuclearReaction.hh"
#endif

// Neutron (for high precision models, user must first download high precision neutron data files from Geant4 web page)
#if (G4VERSION_MAJOR == 9)
#include "G4LCapture.hh"
#include "G4NeutronHPorLCapture.hh"
#include "G4NeutronHPorLElastic.hh"
#include "G4NeutronHPorLEInelastic.hh"
#include "G4NeutronHPorLFission.hh"
#endif

#include "G4NeutronRadCapture.hh" 
#include "G4LFission.hh"
#include "G4NeutronHPCapture.hh"

#include "G4NeutronHPElastic.hh"
#include "G4NeutronHPInelastic.hh"
#include "G4NeutronHPFission.hh"
