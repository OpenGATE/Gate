/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateFissionPB.hh"

#include "GateHadronicStandardProcessMessenger.hh"


//-----------------------------------------------------------------------------
GateFissionPB::GateFissionPB():GateVProcess("Fission")
{  
  SetDefaultParticle("neutron");
  SetProcessInfo("Neutron-induced fission");

  AddToModelList("G4LFission");//default for this process
  AddToDataSetList("G4HadronFissionDataSet"); //default for this process

  // User must first download high precision neutron data files from Geant4 web page
  // For details, see the chapter on the High Precision Neutron Models in the Geant4 Physics Reference Manual.
  AddToModelList("G4NeutronHPFission");
  AddToModelList("G4NeutronHPorLFission");
  AddToDataSetList("G4NeutronHPFissionData"); //default for this process
  //--------------------------------------------------------------------------------

  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateFissionPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4HadronFissionProcess(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFissionPB::ConstructProcess( G4ProcessManager * manager)
{  
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateFissionPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Neutron::Neutron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateFissionPB::IsModelApplicable(G4String ,G4ParticleDefinition *)
{
   return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateFissionPB::IsDatasetApplicable(G4String ,G4ParticleDefinition *)
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFissionPB::AddUserDataSet(G4String ){}
void GateFissionPB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateFissionPB)
