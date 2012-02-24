/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEHADIONLOWEMESSENGER_CC
#define GATEHADIONLOWEMESSENGER_CC

#include "GateHadronIonisationLowEMessenger.hh"


//-----------------------------------------------------------------------------
GateHadronIonisationLowEMessenger::GateHadronIonisationLowEMessenger(GateVProcess *pb):GateEMStandardProcessMessenger(pb)
{
  BuildCommands(pb->GetG4ProcessName() );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
  GateHadronIonisationLowEMessenger::~GateHadronIonisationLowEMessenger()
{
  delete pSetNuclearStopping;
  delete pUnsetNuclearStopping;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationLowEMessenger::BuildCommands(G4String base)
{
  //BuildEMStandardProcessCommands(base);

  BuildModelsCommands(base);

  G4String baseModel = mPrefix+base;

  G4String bb;
  G4String guidance;

  bb = baseModel+"/setNuclearStoppingOn";
  pSetNuclearStopping = new G4UIcmdWithAString(bb,this);
  guidance = "Set nuclear stopping on";
  pSetNuclearStopping->SetGuidance(guidance);
  pSetNuclearStopping->SetParameterName("Particle or Group of particles",   true);
  pSetNuclearStopping->SetDefaultValue("Default");

  bb = baseModel +"/setNuclearStoppingOff";
  pUnsetNuclearStopping = new G4UIcmdWithAString(bb,this);
  guidance = "Set nuclear stopping off";
  pUnsetNuclearStopping->SetGuidance(guidance);
  pUnsetNuclearStopping->SetParameterName("Particle or Group of particles",   true);
  pUnsetNuclearStopping->SetDefaultValue("Default");


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronIonisationLowEMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  GateEMStandardProcessMessenger::SetNewValue(command, param);

  if(command==pAddModel)
    {
      char par1[30];
      char par2[30];
      std::istringstream is(param);
      is >> par1 >> par2 ;
      if(std::string(par1) == "Elec_ICRU_R49p" ||
	 std::string(par1) == "Elec_ICRU_R49He" ||
	 std::string(par1)  == "Elec_Ziegler1977p" ||
	 std::string(par1)  == "Elec_Ziegler1977He"  ||
	 std::string(par1) == "Elec_Ziegler1985p"  ||
	 std::string(par1) == "Elec_Ziegler2000p") {
	pProcess->UnSetModel("Elec_ICRU_R49p", par2);
	pProcess->UnSetModel("Elec_ICRU_R49He", par2);
	pProcess->UnSetModel("Elec_Ziegler1977p", par2);
	pProcess->UnSetModel("Elec_Ziegler1977He", par2);
	pProcess->UnSetModel("Elec_Ziegler1985p", par2);
	pProcess->UnSetModel("Elec_SRIM2000p", par2);
      }
      else if(std::string(par1) == "Nuclear_ICRU_R49" || 
	      std::string(par1) == "Nuclear_Ziegler1977" ||
	      std::string(par1) == "Nuclear_Ziegler1985" ) {
	pProcess->UnSetModel("Nuclear_ICRU_R49", par2);
	pProcess->UnSetModel("Nuclear_Ziegler1977", par2);
	pProcess->UnSetModel("Nuclear_Ziegler1985", par2);
	pProcess->UnSetModel("CalculationOfNuclearStoppingPower_Off", par2);
	pProcess->UnSetModel("CalculationOfNuclearStoppingPower_On", par2);

      }
    }


  SetModelsNewValue(command,param);

  if(command==pUnsetNuclearStopping)
    {
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;

      pProcess->UnSetModel("Nuclear_ICRU_49", par1);
      pProcess->UnSetModel("Nuclear_Ziegler1977", par1);
      pProcess->UnSetModel("Nuclear_Ziegler1985", par1);
      pProcess->UnSetModel("CalculationOfNuclearStoppingPower_Off", par1);
      pProcess->UnSetModel("CalculationOfNuclearStoppingPower_On", par1);
      pProcess->SetModel("CalculationOfNuclearStoppingPower_Off",par1);
    }

  if(command==pSetNuclearStopping)    {
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;

      pProcess->UnSetModel("CalculationOfNuclearStoppingPower_Off", par1);
      pProcess->UnSetModel("CalculationOfNuclearStoppingPower_On", par1);
      pProcess->SetModel("CalculationOfNuclearStoppingPower_On",par1);
  } 

}
//-----------------------------------------------------------------------------

#endif /* end #define GATEHADIONLOWEMESSENGER_CC */
