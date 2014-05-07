/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_ECAT7

#include "GateSinoAccelToEcat7Messenger.hh"
#include "GateSinoAccelToEcat7.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"





GateSinoAccelToEcat7Messenger::GateSinoAccelToEcat7Messenger(GateSinoAccelToEcat7* gateSinoAccelToEcat7)
  : GateOutputModuleMessenger(gateSinoAccelToEcat7)
  , m_gateSinoAccelToEcat7(gateSinoAccelToEcat7)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output ECAT7 sinogram file");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"mashing";
  SetMashingCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetMashingCmd->SetGuidance("Set azimutal mashing factor.");
  SetMashingCmd->SetParameterName("Number",false);
  SetMashingCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"span";
  SetSpanCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetSpanCmd->SetGuidance("Set span (polar mashing) factor.");
  SetSpanCmd->SetParameterName("Number",false);
  SetSpanCmd->SetRange("Number>2");

  cmdName = GetDirectoryName()+"maxringdiff";
  SetMaxRingDiffCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetMaxRingDiffCmd->SetGuidance("Set maximum ring difference.");
  SetMaxRingDiffCmd->SetParameterName("Number",false);
  SetMaxRingDiffCmd->SetRange("Number>0");

  cmdName = GetDirectoryName()+"system";
  SetEcatAccelCameraNumberCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetEcatAccelCameraNumberCmd->SetGuidance("Set camera type according to ECAT numerotation.");
  SetEcatAccelCameraNumberCmd->SetParameterName("Number",false);
  SetEcatAccelCameraNumberCmd->SetRange("Number>0");
}





GateSinoAccelToEcat7Messenger::~GateSinoAccelToEcat7Messenger()
{
  delete SetFileNameCmd;
  delete SetMashingCmd;
  delete SetSpanCmd;
  delete SetMaxRingDiffCmd;
  delete SetEcatAccelCameraNumberCmd;
}





void GateSinoAccelToEcat7Messenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == SetFileNameCmd)
    { m_gateSinoAccelToEcat7->SetFileName(newValue); }
  else if ( command==SetMashingCmd )
    { m_gateSinoAccelToEcat7->SetMashing(SetMashingCmd->GetNewIntValue(newValue)); }
  else if ( command==SetSpanCmd )
    { m_gateSinoAccelToEcat7->SetSpan(SetSpanCmd->GetNewIntValue(newValue)); }
  else if ( command==SetMaxRingDiffCmd )
    { m_gateSinoAccelToEcat7->SetMaxRingDiff(SetMaxRingDiffCmd->GetNewIntValue(newValue)); }
  else if ( command==SetEcatAccelCameraNumberCmd )
    { m_gateSinoAccelToEcat7->SetEcatAccelCameraNumber(SetEcatAccelCameraNumberCmd->GetNewIntValue(newValue)); }
  else
    { GateOutputModuleMessenger::SetNewValue(command,newValue);  }

}
#endif
