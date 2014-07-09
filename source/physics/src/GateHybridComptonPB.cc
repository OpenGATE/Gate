/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHybridComptonPB.hh"
#include "G4HybridComptonProcess.hh"
#include "G4ComptonScattering.hh"
#include "GateEMStandardProcessMessenger.hh"

GateHybridComptonPB::GateHybridComptonPB():GateVProcess("HybridCompton")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Hybrid Compton scattering of gammas");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("LivermorePolarizedModel");
  AddToModelList("PenelopeModel");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateHybridComptonPB::CreateProcess(G4ParticleDefinition *)
{
  G4ComptonScattering* compton = new G4ComptonScattering();
  G4HybridComptonProcess* process = new G4HybridComptonProcess();
  process->RegisterProcess(compton);
  return process;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridComptonPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHybridComptonPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateHybridComptonPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++)
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHybridComptonPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridComptonPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridComptonPB::AddUserModel(GateListOfHadronicModels * model){
  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreComptonModel* theLivermoreComptonModel = new G4LivermoreComptonModel();
#if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
    dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermoreComptonModel);
#else
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreComptonModel);
#endif
  }
  else if(model->GetModelName() == "LivermorePolarizedModel")
  {
    G4LivermorePolarizedComptonModel* theLivermoreComptonModel = new G4LivermorePolarizedComptonModel();
#if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
    dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermoreComptonModel);
#else
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreComptonModel);
#endif
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeComptonModel* theComptonModel = new G4PenelopeComptonModel();
    //    G4WrapperProcess* wpro = dynamic_cast<G4WrapperProcess*>(pProcess);
    G4VEmProcess* vpro = dynamic_cast<G4HybridComptonProcess*>(pProcess)->GetEmProcess();
#if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
    vpro->SetEmModel(theComptonModel);
#else
    vpro->SetModel(theComptonModel);
#endif
  }


}

MAKE_PROCESS_AUTO_CREATOR_CC(GateHybridComptonPB)
