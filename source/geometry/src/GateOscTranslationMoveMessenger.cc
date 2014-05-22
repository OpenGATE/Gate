/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateOscTranslationMoveMessenger.hh"
#include "GateOscTranslationMove.hh"


#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateOscTranslationMoveMessenger::GateOscTranslationMoveMessenger(GateOscTranslationMove* itsTranslationMove)
  :GateObjectRepeaterMessenger(itsTranslationMove)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setAmplitude";
    AmplitudeCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    AmplitudeCmd->SetGuidance("Set the amplitude vector of the oscillating translation.");
    AmplitudeCmd->SetParameterName("Xmax","Ymax","Zmax",false);
    AmplitudeCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName()+"setFrequency";
    FrequencyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    FrequencyCmd->SetGuidance("Set the frequency of the oscillating translation.");
    FrequencyCmd->SetParameterName("frequency",false);
    FrequencyCmd->SetUnitCategory("Frequency");

    cmdName = GetDirectoryName()+"setPeriod";
    PeriodCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    PeriodCmd->SetGuidance("Set the period of the oscillating translation.");
    PeriodCmd->SetParameterName("period",false);
    PeriodCmd->SetUnitCategory("Time");

    cmdName = GetDirectoryName()+"setPhase";
    PhaseCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    PhaseCmd->SetGuidance("Set the phase at t=0 of the oscillating translation.");
    PhaseCmd->SetParameterName("phase",false);
    PhaseCmd->SetUnitCategory("Angle");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateOscTranslationMoveMessenger::~GateOscTranslationMoveMessenger()
{
    delete AmplitudeCmd;
    delete FrequencyCmd;
    delete PeriodCmd;
    delete PhaseCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateOscTranslationMoveMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==AmplitudeCmd )
    { GetTranslationMove()->SetAmplitude(AmplitudeCmd->GetNew3VectorValue(newValue));}      
  else if( command==PeriodCmd )
    { GetTranslationMove()->SetPeriod(PeriodCmd->GetNewDoubleValue(newValue));}      
  else if( command==FrequencyCmd )
    { GetTranslationMove()->SetFrequency(FrequencyCmd->GetNewDoubleValue(newValue));}      
  else if( command==PhaseCmd )
    { GetTranslationMove()->SetPhase(PhaseCmd->GetNewDoubleValue(newValue));}      
  
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
