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

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourceOfPromptGammaMessenger::~GateSourceOfPromptGammaMessenger()
{
  delete pSetFilenameCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceOfPromptGammaMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == pSetFilenameCmd) {
    pSourceOfPromptGamma->SetFilename(newValue);
  }
  // No call to superclass
  // GateVSourceMessenger::SetNewValue(command, newValue);
}
//----------------------------------------------------------------------------------------
