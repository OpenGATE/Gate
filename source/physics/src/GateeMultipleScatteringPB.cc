/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#include "GateeMultipleScatteringPB.hh"
#include "GateMultiScatteringMessenger.hh"

//-----------------------------------------------------------------------------
GateeMultipleScatteringPB::GateeMultipleScatteringPB():GateVProcess("eMultipleScattering")
{  
  SetDefaultParticle("e+"); 
  SetDefaultParticle("e-");

  #if (G4VERSION_MAJOR == 9)
    AddToModelList("Urban95Model");
    AddToModelList("Urban93Model");
  #else
    AddToModelList("UrbanModel"); 
  #endif

  SetProcessInfo("Multiple Coulomb scattering of charged particles");
  pMessenger = new GateMultiScatteringMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateeMultipleScatteringPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4eMultipleScattering(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateeMultipleScatteringPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 1, 1);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateeMultipleScatteringPB::IsApplicable(G4ParticleDefinition * par)
{
  for(unsigned int i=0; i<theListOfDefaultParticles.size(); i++)
      if(par->GetParticleName() == theListOfDefaultParticles[i]) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateeMultipleScatteringPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) 
      GateError("A " << GetG4ProcessName() << " model has been already selected for " << par->GetParticleName());
  if(par == G4Electron::Electron()) return true;
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateeMultipleScatteringPB::AddUserModel(GateListOfHadronicModels *model){
  
  #if (G4VERSION_MAJOR == 9)
    if(model->GetModelName() == "Urban93Model")
      {
	dynamic_cast<G4VMultipleScattering*>(pProcess)->AddEmModel(1, new G4UrbanMscModel95());
      }

    if(model->GetModelName() == "Urban95Model")
      {
	dynamic_cast<G4VMultipleScattering*>(pProcess)->AddEmModel(1, new G4UrbanMscModel95());
      }
  #else
    if(model->GetModelName() == "UrbanModel")
      {
	dynamic_cast<G4VMultipleScattering*>(pProcess)->AddEmModel(1, new G4UrbanMscModel());
      }
  #endif
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateeMultipleScatteringPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Electron::Electron()) return true;
  if(par == G4Positron::Positron()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateeMultipleScatteringPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateeMultipleScatteringPB)
