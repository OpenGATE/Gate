/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGENERALEMPROCESSMESSENGER_CC
#define GATEGENERALEMPROCESSMESSENGER_CC

#include "GateHadronIonIonisationProcessMessenger.hh"

//-----------------------------------------------------------------------------
GateHadronIonIonisationProcessMessenger::GateHadronIonIonisationProcessMessenger(GateVProcess *pb):GateEMStandardProcessMessenger(pb)
{
  BuildCommands(pb->GetG4ProcessName() );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHadronIonIonisationProcessMessenger::~GateHadronIonIonisationProcessMessenger()
{
  delete pSetNuclearStopping;
  delete pUnsetNuclearStopping;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHadronIonIonisationProcessMessenger::BuildCommands(G4String base)
{

  G4String baseModel ="";

 baseModel +=  mPrefix;
  baseModel += base;


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
void GateHadronIonIonisationProcessMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  GateEMStandardProcessMessenger::SetNewValue(command, param);

  if(command==pUnsetNuclearStopping){
      char par1[30];
      std::istringstream is(param);
      is >> par1 ;

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

#endif /* end #define GATEGENRALEMPROCESSMESSENGER_CC */
