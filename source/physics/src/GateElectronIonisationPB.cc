/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#include "GateElectronIonisationPB.hh"

#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateElectronIonisationPB::GateElectronIonisationPB():GateVProcess("ElectronIonisation")
{  
  SetDefaultParticle("e+");
  SetDefaultParticle("e-");
  SetProcessInfo("Ionization and energy loss by electrons and positrons");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("PenelopeModel");


  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateElectronIonisationPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4eIonisation(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateElectronIonisationPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateElectronIonisationPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateElectronIonisationPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateElectronIonisationPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron() || par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateElectronIonisationPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateElectronIonisationPB::AddUserModel(GateListOfHadronicModels * model){

  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreIonisationModel* theLivermoreIonisationModel = new G4LivermoreIonisationModel();
    dynamic_cast<G4VEnergyLossProcess*>(pProcess)->SetEmModel(theLivermoreIonisationModel);
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeIonisationModel* theIonisationModel = new G4PenelopeIonisationModel();
    dynamic_cast<G4VEnergyLossProcess*>(pProcess)->SetEmModel(theIonisationModel);
  }
}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GateElectronIonisationPB)

