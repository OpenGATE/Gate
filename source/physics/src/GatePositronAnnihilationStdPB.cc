/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#include "GateConfiguration.h"

#include "GatePositronAnnihilationStdPB.hh"
#include "GateEMStandardProcessMessenger.hh"


#ifdef G4VERSION9_3
//-----------------------------------------------------------------------------
GatePositronAnnihilationStdPB::GatePositronAnnihilationStdPB():GateVProcess("G4PositronAnnihilation")
{  
  SetDefaultParticle("e+");
  SetProcessInfo("Standard G4 positron annihilation process");

  AddToModelList("StandardModel");
  AddToModelList("PenelopeModel");

  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePositronAnnihilationStdPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4eplusAnnihilation();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePositronAnnihilationStdPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),0, -1, 4); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePositronAnnihilationStdPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePositronAnnihilationStdPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePositronAnnihilationStdPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePositronAnnihilationStdPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePositronAnnihilationStdPB::AddUserModel(GateListOfHadronicModels *model){

  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeAnnihilationModel* theAnnihilationModel = new G4PenelopeAnnihilationModel();
    dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theAnnihilationModel);
  }
}
//-----------------------------------------------------------------------------


#else
//-----------------------------------------------------------------------------
GatePositronAnnihilationStdPB::GatePositronAnnihilationStdPB():GateVProcess("PositronAnnihilationStd")
{  
  SetDefaultParticle("e+");
  SetProcessInfo("Standard G4 positron annihilation process");
  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePositronAnnihilationStdPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4eplusAnnihilation();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePositronAnnihilationStdPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),0, -1, 4); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePositronAnnihilationStdPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------
#endif


MAKE_PROCESS_AUTO_CREATOR_CC(GatePositronAnnihilationStdPB)

