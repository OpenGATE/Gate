/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateSourceOfPromptGammaMessenger.hh"
#include "GateSourceOfPromptGamma.hh"

//----------------------------------------------------------------------------------------
GateSourceOfPromptGammaMessenger::
GateSourceOfPromptGammaMessenger(GateSourceOfPromptGamma* source)
:GateVSourceMessenger(source)
{
  pSourceOfPromptGamma = source;
  G4String cmdName;

  // Set Filename command
  G4String s = GetDirectoryName()+"setFilename";
  pSetFilenameCmd = new G4UIcmdWithAString(s,this);
  pSetFilenameCmd->SetGuidance("Set filename that contains the 3D spectrum distribution");

  s = GetDirectoryName()+"setTof";
  pSetTofCmd = new G4UIcmdWithABool(s, this);
  pSetTofCmd->SetGuidance("Set use of Time of Flight");  
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourceOfPromptGammaMessenger::~GateSourceOfPromptGammaMessenger()
{
  delete pSetFilenameCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceOfPromptGammaMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == pSetFilenameCmd) pSourceOfPromptGamma->SetFilename(newValue);
  if (command == pSetTofCmd) pSourceOfPromptGamma->SetTof(pSetTofCmd->GetNewBoolValue(newValue));
  
  // No call to superclass
  // GateVSourceMessenger::SetNewValue(command, newValue);
}
//----------------------------------------------------------------------------------------
