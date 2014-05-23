/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHadronIonisationPB.hh"

#include "GateHadronIonIonisationProcessMessenger.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//-----------------------------------------------------------------------------
GateHadronIonisationPB::GateHadronIonisationPB():GateVProcess("HadronIonisation")
{  
  SetDefaultParticle("pi+"); SetDefaultParticle("pi-");
  SetDefaultParticle("kaon+"); SetDefaultParticle("kaon-");
  SetDefaultParticle("sigma+"); SetDefaultParticle("sigma-");
  SetDefaultParticle("proton"); SetDefaultParticle("anti_proton");
  SetDefaultParticle("xi-"); SetDefaultParticle("anti_xi-");
  SetDefaultParticle("anti_sigma+"); SetDefaultParticle("anti_sigma-");
  SetDefaultParticle("omega-"); SetDefaultParticle("anti_omega-");
  SetDefaultParticle("deuteron");
  SetDefaultParticle("triton");
  SetDefaultParticle("He3");
  SetDefaultParticle("alpha");
  SetDefaultParticle("GenericIon");

  SetProcessInfo("Ionization and energy loss by charged hadrons");
  pMessenger = new GateHadronIonIonisationProcessMessenger(this);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess* GateHadronIonisationPB::CreateProcess(G4ParticleDefinition *)
{
 return new G4hIonisation(GetG4ProcessName()); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);   
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronIonisationPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par->GetPDGCharge() != 0.0 && par->GetPDGMass() > 10.0*MeV && !par->IsShortLived()) return true;
  return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateHadronIonisationPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  if( par->GetPDGCharge() != 0.0 && par->GetPDGMass() > proton_mass_c2*0.1 )  return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronIonisationPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationPB::AddUserModel(GateListOfHadronicModels * /*model*/)
{
  /* if(model->GetModelName() == "CalculationOfNuclearStoppingPower_On"){
      dynamic_cast<G4hIonisation*>(GetUserModelProcess())->ActivateNuclearStopping(true);
          
  }
  else if(model->GetModelName() == "CalculationOfNuclearStoppingPower_Off"){
      dynamic_cast<G4hIonisation*>(GetUserModelProcess())->ActivateNuclearStopping(false);

  }*/
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------



MAKE_PROCESS_AUTO_CREATOR_CC(GateHadronIonisationPB)
