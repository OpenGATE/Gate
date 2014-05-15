/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateGammaConversionPB.hh"
#include "GateConfiguration.h"
#include "GateEMStandardProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateGammaConversionPB::GateGammaConversionPB():GateVProcess("GammaConversion")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Pair production by gammas");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("LivermorePolarizedModel");
  AddToModelList("PenelopeModel");

  pMessenger = new GateEMStandardProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateGammaConversionPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4GammaConversion(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGammaConversionPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateGammaConversionPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateGammaConversionPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateGammaConversionPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGammaConversionPB::AddUserDataSet(G4String ){}
void GateGammaConversionPB::AddUserModel(GateListOfHadronicModels *model){

  if(model->GetModelName() == "StandardModel")
  {
  }
  else if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermoreGammaConversionModel* theLivermoreGammaConversionModel = new G4LivermoreGammaConversionModel();
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermoreGammaConversionModel); 
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreGammaConversionModel); 
    #endif
  }
  else if(model->GetModelName() == "LivermorePolarizedModel")
  {
    G4LivermorePolarizedGammaConversionModel* theLivermoreGammaConversionModel = new G4LivermorePolarizedGammaConversionModel();
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
    dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermoreGammaConversionModel);  
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermoreGammaConversionModel); 
    #endif
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopeGammaConversionModel* theGammaConversionModel = new G4PenelopeGammaConversionModel();
    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
    dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theGammaConversionModel);  
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theGammaConversionModel); 
    #endif
  }
}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GateGammaConversionPB)
