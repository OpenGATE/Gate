/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#include "GateConfiguration.h"

#include "GateBremsstrahlungPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateBremsstrahlungPB::GateBremsstrahlungPB():GateVProcess("Bremsstrahlung")
{  
  SetDefaultParticle("e+");
  SetDefaultParticle("e-");
  SetProcessInfo("Bremsstrahlung by electrons and positrons");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("PenelopeModel");

  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateBremsstrahlungPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4eBremsstrahlung(GetG4ProcessName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateBremsstrahlungPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(pFinalProcess,-1, -3, 3);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateBremsstrahlungPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateBremsstrahlungPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateBremsstrahlungPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateBremsstrahlungPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateBremsstrahlungPB::AddUserModel(GateListOfHadronicModels *model){

  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreBremsstrahlungModel* theLivermoreBremsstrahlungModel = new G4LivermoreBremsstrahlungModel();
    dynamic_cast<G4VEnergyLossProcess*>(pProcess)->SetEmModel(theLivermoreBremsstrahlungModel);
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeBremsstrahlungModel* theBremsstrahlungModel = new G4PenelopeBremsstrahlungModel();
    dynamic_cast<G4VEnergyLossProcess*>(pProcess)->SetEmModel(theBremsstrahlungModel);
  }


}
//-----------------------------------------------------------------------------



MAKE_PROCESS_AUTO_CREATOR_CC(GateBremsstrahlungPB)

