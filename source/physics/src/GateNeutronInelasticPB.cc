/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateNeutronInelasticPB.hh"

#include "GateHadronicStandardProcessMessenger.hh"


//!!!!!!!!!! No model for High Energy !!!!!!!!!!!!!

//-----------------------------------------------------------------------------
GateNeutronInelasticPB::GateNeutronInelasticPB():GateVProcess("NeutronInelastic")
{  
  SetDefaultParticle("neutron");
  SetProcessInfo("Inelastic scattering of neutrons");

  AddToModelList("G4LENeutronInelastic");
  AddToModelList("G4BertiniCascade");
  AddToModelList("G4BinaryCascade");
  AddToModelList("GateBinaryCascade");
  AddToModelList("PreCompound");
  AddToModelList("LeadingParticleBias");

  AddToDataSetList("G4HadronInelasticDataSet"); // default for this process
  AddToDataSetList("G4NeutronInelasticCrossSection");

  // User must first download high precision neutron data files from Geant4 web page
  // For details, see the chapter on the High Precision Neutron Models in the Geant4 Physics Reference Manual.
  AddToModelList("G4NeutronHPInelastic");
  AddToModelList("G4NeutronHPorLEInelastic");

  AddToDataSetList("G4NeutronHPInelasticData");
  //--------------------------------------------------------------------------------

  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateNeutronInelasticPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4NeutronInelasticProcess(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronInelasticPB::ConstructProcess( G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateNeutronInelasticPB::IsApplicable(G4ParticleDefinition * par)
{
   if(par == G4Neutron::Neutron()) return true;
   return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateNeutronInelasticPB::IsModelApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateNeutronInelasticPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
   return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronInelasticPB::AddUserDataSet(G4String ){}
void GateNeutronInelasticPB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------

MAKE_PROCESS_AUTO_CREATOR_CC(GateNeutronInelasticPB)
