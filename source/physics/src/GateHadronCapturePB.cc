/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHadronCapturePB.hh"

#include "GateHadronicStandardProcessMessenger.hh"


//-----------------------------------------------------------------------------
GateHadronCapturePB::GateHadronCapturePB():GateVProcess("NeutronCapture")
{  
  SetDefaultParticle("neutron");
  SetProcessInfo("Neutron capture");

  AddToModelList("G4LCapture");
  AddToDataSetList("G4HadronCaptureDataSet"); //default for this process

  // User must first download high precision neutron data files from Geant4 web page
  // For details, see the chapter on the High Precision Neutron Models in the Geant4 Physics Reference Manual.
  AddToModelList("G4NeutronHPCapture"); 
  AddToModelList("G4NeutronHPorLCapture");
  AddToDataSetList("G4NeutronHPCaptureData");
  //--------------------------------------------------------------------------------

  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateHadronCapturePB::CreateProcess(G4ParticleDefinition *)
{
  return new G4HadronCaptureProcess(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronCapturePB::ConstructProcess( G4ProcessManager * manager)
{  
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronCapturePB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Neutron::Neutron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronCapturePB::IsModelApplicable(G4String ,G4ParticleDefinition * )
{
   return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronCapturePB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronCapturePB::AddUserDataSet(G4String ){}
void GateHadronCapturePB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateHadronCapturePB)
