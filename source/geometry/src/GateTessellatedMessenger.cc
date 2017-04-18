/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateTessellatedMessenger.hh"
#include "GateTessellated.hh"

GateTessellatedMessenger::GateTessellatedMessenger(
  GateTessellated* itsCreator )
: GateVolumeMessenger( itsCreator )
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName;
  cmdName = dir + "setPathToVerticesFile";
  PathToVerticesFileCmd = new G4UIcmdWithAString( cmdName, this );
  PathToVerticesFileCmd->SetGuidance( "Set path to raw file" );
}

GateTessellatedMessenger::~GateTessellatedMessenger()
{
  delete PathToVerticesFileCmd;
}

void GateTessellatedMessenger::SetNewValue( G4UIcommand* command,
  G4String newValue )
{
  if( command == PathToVerticesFileCmd )
  {
    GetTessellatedCreator()->SetPathToVerticesFile( newValue );
  }
  else
  {
    GateVolumeMessenger::SetNewValue( command, newValue );
  }
}