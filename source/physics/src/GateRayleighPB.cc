/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"


#include "GateRayleighPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateRayleighPB::GateRayleighPB():GateVProcess("RayleighScattering")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Rayleigh scattering of gammas");

  AddToModelList("LivermoreModel");
  AddToModelList("LivermorePolarizedModel");
  AddToModelList("PenelopeModel");


  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateRayleighPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4RayleighScattering(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRayleighPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateRayleighPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateRayleighPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateRayleighPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRayleighPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRayleighPB::AddUserModel(GateListOfHadronicModels *model){
  if(model->GetModelName() == "StandardModel")
  {
    // Default one
  }
  if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreRayleighModel* theLivermoreRayleighModel = new G4LivermoreRayleighModel();
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5)) 
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermoreRayleighModel); 
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreRayleighModel); 
    #endif
  }
  else if(model->GetModelName() == "LivermorePolarizedModel")
  {
    G4LivermorePolarizedRayleighModel* theLivermoreRayleighModel = new G4LivermorePolarizedRayleighModel();
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5)) 
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermoreRayleighModel);
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreRayleighModel); 
    #endif
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeRayleighModel* theRayleighModel = new G4PenelopeRayleighModel();
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5)) 
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theRayleighModel); 
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theRayleighModel); 
    #endif
  }


}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateRayleighPB)
