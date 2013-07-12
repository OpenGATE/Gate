/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#include "GateComptonPB.hh"

#include "GateEMStandardProcessMessenger.hh"


//-----------------------------------------------------------------------------
GateComptonPB::GateComptonPB():GateVProcess("Compton")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Compton scattering of gammas");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("LivermorePolarizedModel");
  AddToModelList("PenelopeModel");

  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateComptonPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4ComptonScattering(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateComptonPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateComptonPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateComptonPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonPB::AddUserModel(GateListOfHadronicModels * model){
  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreComptonModel* theLivermoreComptonModel = new G4LivermoreComptonModel();
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreComptonModel);
  }
  else if(model->GetModelName() == "LivermorePolarizedModel")
  {
    G4LivermorePolarizedComptonModel* theLivermoreComptonModel = new G4LivermorePolarizedComptonModel();
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreComptonModel);
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeComptonModel* theComptonModel = new G4PenelopeComptonModel();
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theComptonModel);
  }


}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GateComptonPB)
