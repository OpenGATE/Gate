/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateSourcePromptGammaEmissionMessenger.hh"
#include "GateSourcePromptGammaEmission.hh"

//----------------------------------------------------------------------------------------
GateSourcePromptGammaEmissionMessenger::
GateSourcePromptGammaEmissionMessenger(GateSourcePromptGammaEmission* source)
:GateVSourceMessenger(source)
{
  pSourcePromptGammaEmission = source;
  G4String cmdName;

  // Set Filename command
  G4String s = GetDirectoryName()+"setFilename";
  pSetFilenameCmd = new G4UIcmdWithAString(s,this);
  pSetFilenameCmd->SetGuidance("Set filename that contains the 3D spectrum distribution");

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourcePromptGammaEmissionMessenger::~GateSourcePromptGammaEmissionMessenger()
{
  delete pSetFilenameCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourcePromptGammaEmissionMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == pSetFilenameCmd) {
    pSourcePromptGammaEmission->SetFilename(newValue);
  }
  // No call to superclass
  // GateVSourceMessenger::SetNewValue(command, newValue);
}
//----------------------------------------------------------------------------------------
