/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHybridRayleighPB.hh"
#include "G4HybridRayleighProcess.hh"
#include "G4RayleighScattering.hh"
#include "GateEMStandardProcessMessenger.hh"

GateHybridRayleighPB::GateHybridRayleighPB():GateVProcess("HybridRayleigh")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Hybrid Rayleigh scattering of gammas");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("LivermorePolarizedModel");
  AddToModelList("PenelopeModel");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateHybridRayleighPB::CreateProcess(G4ParticleDefinition *)
{
  G4RayleighScattering* compton = new G4RayleighScattering();
  G4HybridRayleighProcess* process = new G4HybridRayleighProcess();
  process->RegisterProcess(compton);
  return process;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridRayleighPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHybridRayleighPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateHybridRayleighPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHybridRayleighPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridRayleighPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridRayleighPB::AddUserModel(GateListOfHadronicModels * model){
  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreRayleighModel* theLivermoreRayleighModel = new G4LivermoreRayleighModel();
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreRayleighModel);
  }
  else if(model->GetModelName() == "LivermorePolarizedModel")
  {
    G4LivermorePolarizedRayleighModel* theLivermoreRayleighModel = new G4LivermorePolarizedRayleighModel();
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreRayleighModel);
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeRayleighModel* theRayleighModel = new G4PenelopeRayleighModel();
    //    G4WrapperProcess* wpro = dynamic_cast<G4WrapperProcess*>(pProcess);
    G4VEmProcess* vpro = dynamic_cast<G4HybridRayleighProcess*>(pProcess)->GetEmProcess();
    vpro->SetModel(theRayleighModel);
  }


}

MAKE_PROCESS_AUTO_CREATOR_CC(GateHybridRayleighPB)
