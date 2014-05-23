/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePionMinusInelasticPB.hh"

#include "GateHadronicStandardProcessMessenger.hh"


//!!!!!!!!!! No model for High Energy !!!!!!!!!!!!!

//-----------------------------------------------------------------------------
GatePionMinusInelasticPB::GatePionMinusInelasticPB():GateVProcess("PionMinusInelastic")
{  
  SetDefaultParticle("pi-");
  SetProcessInfo("Inelastic scattering of pi+");

  AddToModelList("G4LEPionMinusInelastic");
  AddToModelList("G4BertiniCascade");
  AddToModelList("G4BinaryCascade");
  AddToModelList("LeadingParticleBias");

  AddToDataSetList("G4HadronInelasticDataSet"); // default for this process
  AddToDataSetList("G4PiNuclearCrossSection");


  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePionMinusInelasticPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4PionMinusInelasticProcess(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePionMinusInelasticPB::ConstructProcess( G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePionMinusInelasticPB::IsApplicable(G4ParticleDefinition * par)
{
   if(par == G4PionMinus::PionMinus()) return true;
   return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePionMinusInelasticPB::IsModelApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePionMinusInelasticPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePionMinusInelasticPB::AddUserDataSet(G4String ){}
void GatePionMinusInelasticPB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GatePionMinusInelasticPB)
