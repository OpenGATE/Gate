/*----------------------
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
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))  
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theComptonModel); 
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theComptonModel); 
    #endif
  }


}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GateComptonPB)
