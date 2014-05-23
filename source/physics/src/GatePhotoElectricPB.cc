/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateConfiguration.h"
#include "GatePhotoElectricPB.hh"
#include "GatePhotoElectricMessenger.hh"


//-----------------------------------------------------------------------------
GatePhotoElectricPB::GatePhotoElectricPB():GateVProcess("PhotoElectric")
{  
  SetDefaultParticle("gamma");
  SetProcessInfo("Photo electric effect");

  AddToModelList("StandardModel");
  AddToModelList("LivermoreModel");
  AddToModelList("LivermorePolarizedModel");
  AddToModelList("PenelopeModel");

  pMessenger = new GatePhotoElectricMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GatePhotoElectricPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4PhotoElectricEffect(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhotoElectricPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddDiscreteProcess(GetProcess());           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePhotoElectricPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePhotoElectricPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  for(unsigned int k = 0; k<theListOfParticlesWithSelectedModels.size();k++) 
    if(par==theListOfParticlesWithSelectedModels[k]) GateError("A "<< GetG4ProcessName()<<" model has been already selected for "<< par->GetParticleName());
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GatePhotoElectricPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par == G4Gamma::Gamma()) return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhotoElectricPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhotoElectricPB::AddUserModel(GateListOfHadronicModels *model){
  if(model->GetModelName() == "StandardModel")
  {
    // default one
  }

  if(model->GetModelName() == "LivermoreModel")
  {
    G4LivermorePhotoElectricModel* theLivermorePhotoElectricModel = new G4LivermorePhotoElectricModel();

    //bool auger = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetIsAugerActivated();
    //double lowEGamma = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyGammaCut();
    //double lowEElec = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyElectronCut();
    //if(auger)theLivermorePhotoElectricModel ->ActivateAuger(true);
    // if( lowEGamma>0 ) theLivermorePhotoElectricModel->SetCutForLowEnSecPhotons(lowEGamma);
    // if( lowEElec>0  ) theLivermorePhotoElectricModel->SetCutForLowEnSecElectrons(lowEElec);

    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermorePhotoElectricModel); 
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermorePhotoElectricModel); 
    #endif
  }
  else if(model->GetModelName() == "LivermorePolarizedModel")
  {
    G4LivermorePolarizedPhotoElectricModel* theLivermorePhotoElectricModel = new G4LivermorePolarizedPhotoElectricModel();

    //bool auger = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetIsAugerActivated();
    //double lowEGamma = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyGammaCut();
    //double lowEElec = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyElectronCut();
    //if(auger) theLivermorePhotoElectricModel->ActivateAuger(true);
    // if( lowEGamma>0 ) theLivermorePhotoElectricModel->SetCutForLowEnSecPhotons(lowEGamma);
    // if( lowEElec>0  ) theLivermorePhotoElectricModel->SetCutForLowEnSecElectrons(lowEElec);

    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
       dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(theLivermorePhotoElectricModel); 
     #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(theLivermorePhotoElectricModel); 
    #endif
  }
  else if(model->GetModelName() == "PenelopeModel")
  {
    G4PenelopePhotoElectricModel* thePhotoElectricModel = new G4PenelopePhotoElectricModel();

    //bool auger = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetIsAugerActivated();
    //double lowEGamma = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyGammaCut();
    //double lowEElec = dynamic_cast<GatePhotoElectricMessenger*>(pMessenger)->GetLowEnergyElectronCut();
    //if(auger) thePhotoElectricModel->ActivateAuger(true);
    //if( lowEGamma>0 )  GateWarning("'setXRayCut' option is not available with standard model");
    //if( lowEElec>0  )  GateWarning("'setDeltaRayCut' option is not available with standard model");

    //if( lowEGamma>0 ) thePhotoElectricModel->SetCutForLowEnSecPhotons(lowEGamma);
    //if( lowEElec>0  ) thePhotoElectricModel->SetCutForLowEnSecElectrons(lowEElec);

    #if (G4VERSION_MAJOR > 9) || ((G4VERSION_MAJOR ==9 && G4VERSION_MINOR > 5))
    dynamic_cast<G4VEmProcess*>(pProcess)->SetEmModel(thePhotoElectricModel); 
    #else
       dynamic_cast<G4VEmProcess*>(pProcess)->SetModel(thePhotoElectricModel); 
    #endif
  }
}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GatePhotoElectricPB)
