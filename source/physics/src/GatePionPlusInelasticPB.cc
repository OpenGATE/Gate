/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePionPlusInelasticPB.hh"

#include "GateHadronicStandardProcessMessenger.hh"


//!!!!!!!!!! No model for High Energy !!!!!!!!!!!!!

//-----------------------------------------------------------------------------
GatePionPlusInelasticPB::GatePionPlusInelasticPB():GateVProcess("PionPlusInelastic")
{  
  SetDefaultParticle("pi+");
  SetProcessInfo("Inelastic scattering of pi+");

  AddToModelList("G4LEPionPlusInelastic");
  AddToModelList("G4BertiniCascade");
  AddToModelList("G4BinaryCascade");
  AddToModelList("LeadingParticleBias");

  AddToDataSetList("G4HadronInelasticDataSet"); // default for this process
  AddToDataSetList("G4PiNuclearCrossSection");


  pMessenger = new GateHadronicStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePionPlusInelasticPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4PionPlusInelasticProcess(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePionPlusInelasticPB::ConstructProcess( G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePionPlusInelasticPB::IsApplicable(G4ParticleDefinition * par)
{
   if(par == G4PionPlus::PionPlus()) return true;
   return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePionPlusInelasticPB::IsModelApplicable(G4String ,G4ParticleDefinition * )
{
   return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePionPlusInelasticPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePionPlusInelasticPB::AddUserDataSet(G4String ){}
void GatePionPlusInelasticPB::AddUserModel(GateListOfHadronicModels *){}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GatePionPlusInelasticPB)
