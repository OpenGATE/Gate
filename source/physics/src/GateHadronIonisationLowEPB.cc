/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHadronIonisationLowEPB.hh"
#include "GateHadronIonisationLowEMessenger.hh"


//-----------------------------------------------------------------------------
GateHadronIonisationLowEPB::GateHadronIonisationLowEPB():GateVProcess("LowEnergyHadronIonisation")
{  
  SetDefaultParticle("mu+"); SetDefaultParticle("mu-");
  SetDefaultParticle("tau+"); SetDefaultParticle("tau-");
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


  //AddToModelList("No_Calculation_Of_Nuclear_Stopping_Power");

  AddToModelList("Elec_ICRU_R49p");
  AddToModelList("Elec_ICRU_R49He");
  AddToModelList("Elec_Ziegler1977p");
  AddToModelList("Elec_Ziegler1977He");
  AddToModelList("Elec_Ziegler1985p");
  AddToModelList("Elec_SRIM2000p");
  AddToModelList("Nuclear_ICRU_R49");
  AddToModelList("Nuclear_Ziegler1977");
  AddToModelList("Nuclear_Ziegler1985");

  SetProcessInfo("Ionization and energy loss by charged hadrons, muons and taus");

  pMessenger = new GateHadronIonisationLowEMessenger(this);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4VProcess*GateHadronIonisationLowEPB::CreateProcess(G4ParticleDefinition *)
{
  return new G4hLowEnergyIonisation(GetG4ProcessName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationLowEPB::ConstructProcess(G4ProcessManager * manager)
{
  manager->AddProcess(GetProcess(),-1, 2, 2);           
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronIonisationLowEPB::IsApplicable(G4ParticleDefinition * par)
{
  if(par->GetPDGCharge() != 0.0 && par->GetPDGMass() > proton_mass_c2*0.1)
  {
    SetModel("Elec_ICRU_R49p",par->GetParticleName());
    SetModel("Nuclear_ICRU_R49",par->GetParticleName());
    return true;
  }
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronIonisationLowEPB::IsModelApplicable(G4String ,G4ParticleDefinition * par)
{
  if( par->GetPDGCharge() != 0.0 && par->GetPDGMass() > proton_mass_c2*0.1 )  return true;
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateHadronIonisationLowEPB::IsDatasetApplicable(G4String ,G4ParticleDefinition * )
{
  return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationLowEPB::AddUserModel(GateListOfHadronicModels *model)
{
  if(model->GetModelName() == "CalculationOfNuclearStoppingPower_On"){
	 // dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())->ActivateNuclearStopping(true);
          dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())-> SetNuclearStoppingOn();
  }
  else if(model->GetModelName() == "Elec_ICRU_R49p" || model->GetModelName() == "Elec_ICRU_R49He"
     || model->GetModelName() == "Elec_Ziegler1977p" || model->GetModelName() == "Elec_Ziegler1977He"
     || model->GetModelName() == "Elec_Ziegler1985p" || model->GetModelName() == "Elec_SRIM2000p" )
  {
   for(unsigned int j=0; j<theListOfSelectedModels.size(); j++)
      if(theListOfSelectedModels[j]==model)
            dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())->SetElectronicStoppingPowerModel(theListOfParticlesWithSelectedModels[j], model->GetModelName().remove(0, 5));
  }
  else if(model->GetModelName() == "Nuclear_ICRU_R49" 
          || model->GetModelName() == "Nuclear_Ziegler1977" 
          || model->GetModelName() == "Nuclear_Ziegler1985" )
  {
     dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())->SetNuclearStoppingPowerModel(model->GetModelName().remove(0, 8));
     dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())->SetNuclearStoppingOn();
  }
  else if(model->GetModelName() == "CalculationOfNuclearStoppingPower_Off"){
          dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())-> SetNuclearStoppingOff();
	 // dynamic_cast<G4hLowEnergyIonisation*>(GetUserModelProcess())->ActivateNuclearStopping(false);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationLowEPB::AddUserDataSet(G4String ){}
//-----------------------------------------------------------------------------


MAKE_PROCESS_AUTO_CREATOR_CC(GateHadronIonisationLowEPB)

