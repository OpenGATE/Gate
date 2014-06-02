/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateIonIonisationPB.hh"
#include "GateHadronIonIonisationProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateIonIonisationPB::GateIonIonisationPB():GateVProcess("IonIonisation")
{  
  SetDefaultParticle("GenericIon");
  SetDefaultParticle("alpha");
  SetDefaultParticle("deuteron");
  SetDefaultParticle("triton");
  SetDefaultParticle("He3");

  SetProcessInfo("Ionization and energy loss by ions");
  pMessenger = new GateHadronIonIonisationProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateIonIonisationPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4ionIonisation(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIonIonisationPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateIonIonisationPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par->GetPDGCharge() != 0.0 && !par->IsShortLived() &&
          par->GetParticleType() == "nucleus") return true;
  return false;
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateIonIonisationPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  if(par->GetPDGCharge() != 0.0 && !par->IsShortLived() &&
          par->GetParticleType() == "nucleus")  return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateIonIonisationPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIonIonisationPB::AddUserModel(GateListOfHadronicModels * /*model*/)
{
/*  if(model->GetModelName() == "CalculationOfNuclearStoppingPower_On"){
      dynamic_cast<G4ionIonisation*>(GetUserModelProcess())->ActivateNuclearStopping(true);
          
  }
  else if(model->GetModelName() == "CalculationOfNuclearStoppingPower_Off"){
      dynamic_cast<G4ionIonisation*>(GetUserModelProcess())->ActivateNuclearStopping(false);

  }*/
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIonIonisationPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------




MAKE_PROCESS_AUTO_CREATOR_CC(GateIonIonisationPB)

