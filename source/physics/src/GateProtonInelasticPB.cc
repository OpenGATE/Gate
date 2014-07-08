/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateProtonInelasticPB.hh"

//#include "G4EnergyRangeManager.hh"
//#include "G4HadronicInteraction.hh"

#include "GateHadronicStandardProcessMessenger.hh"


//!!!!!!!!!! No model for High Energy !!!!!!!!!!!!!

//-----------------------------------------------------------------------------
GateProtonInelasticPB::GateProtonInelasticPB():GateVProcess("ProtonInelastic")
{  
  SetDefaultParticle("proton");
  SetProcessInfo("Inelastic scattering of protons");

  AddToModelList("G4LEProtonInelastic");
  AddToModelList("G4BertiniCascade");
  AddToModelList("G4BinaryCascade");
  AddToModelList("GateBinaryCascade"); 
  AddToModelList("PreCompound");
  AddToModelList("LeadingParticleBias");
  AddToModelList("G4QMDReaction");

  AddToDataSetList("G4HadronInelasticDataSet"); // default for this process
  AddToDataSetList("G4ProtonInelasticCrossSection");


  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateProtonInelasticPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4ProtonInelasticProcess(GetG4ProcessName());
  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateProtonInelasticPB::ConstructProcess( G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateProtonInelasticPB::IsApplicable(G4ParticleDefinition * par)
{
   if(par == G4Proton::Proton()) return true;
   return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateProtonInelasticPB::IsModelApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateProtonInelasticPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateProtonInelasticPB::AddUserDataSet(G4String ){}
void GateProtonInelasticPB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GateProtonInelasticPB)
