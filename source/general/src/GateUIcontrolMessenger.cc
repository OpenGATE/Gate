/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateUIcontrolMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "G4UImanager.hh"
#include "G4UIbatch.hh"

#include "GateTools.hh"


// Constructor
GateUIcontrolMessenger::GateUIcontrolMessenger()
: GateMessenger("control")
{ 
  GetDirectory()->SetGuidance("GATE user-interface commands");

  G4String cmdName;

  cmdName = GetDirectoryName()+"execute";
  ExecuteCmd = new G4UIcmdWithAString(cmdName,this);
  ExecuteCmd->SetGuidance("This command, similar to '/control/execute', execute a macro file");
  ExecuteCmd->SetGuidance("However, if the macro file can not be found in the current directory, ");
  ExecuteCmd->SetGuidance("it tries to find it in the directory $GATEHOME");
  ExecuteCmd->SetParameterName("fileName",false);

}





GateUIcontrolMessenger::~GateUIcontrolMessenger()
{
  delete ExecuteCmd; 
}



// UI command interpreter method
void GateUIcontrolMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==ExecuteCmd )
    { LaunchMacroFile(newValue); }   
  else
    GateMessenger::SetNewValue(command,newValue);
}



// Execute a macrofile located either in the current directory or in $GATEHOME
void GateUIcontrolMessenger::LaunchMacroFile(G4String fileName)
{
  G4String filePath = GateTools::FindGateFile(fileName);
  if (filePath.empty()) {
    G4cerr << "Could not find macro file '" << fileName << "'! Ignored!" << G4endl;
    return;
  }

  G4UImanager::GetUIpointer()->ExecuteMacroFile(filePath);
}

