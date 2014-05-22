/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


// Elastic
#include "G4HadronElasticDataSet.hh"

// Inelastic
#include "G4HadronInelasticDataSet.hh"
#include "G4ProtonInelasticCrossSection.hh"
#include "G4PiNuclearCrossSection.hh"

// ion cross sections:
#include "G4TripathiCrossSection.hh"
#include "G4IonsShenCrossSection.hh"
#include "G4IonsKoxCrossSection.hh"
#include "G4IonsSihverCrossSection.hh"
#include "G4TripathiLightCrossSection.hh"

// Neutron
//#include "G4HadronCaptureDataSet.hh"
#include "G4NeutronInelasticCrossSection.hh"
  // User must first download high precision neutron data files from Geant4 web page
  // For details, see the chapter on the High Precision Neutron Models in the Geant4 Physics Reference Manual.
#include "G4NeutronHPInelasticData.hh"
#include "G4NeutronHPFissionData.hh"
#include "G4NeutronHPElasticData.hh"

// Fission
//#include "G4HadronFissionDataSet.hh"

// Photo-Nuclear
//#include "G4PhotoNuclearCrossSection.hh"

