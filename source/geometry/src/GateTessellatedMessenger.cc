/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateTessellatedMessenger.hh"
#include "GateTessellated.hh"

GateTessellatedMessenger::GateTessellatedMessenger(
  GateTessellated* itsCreator )
: GateVolumeMessenger( itsCreator )
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName;
  cmdName = dir + "setPathToSTLFile";
  PathToSTLFileCmd = new G4UIcmdWithAString( cmdName, this );
  PathToSTLFileCmd->SetGuidance( "Set path to STL file" );
}

GateTessellatedMessenger::~GateTessellatedMessenger()
{
  delete PathToSTLFileCmd;
}

void GateTessellatedMessenger::SetNewValue( G4UIcommand* command,
  G4String newValue )
{
  if( command == PathToSTLFileCmd )
  {
    GetTessellatedCreator()->SetPathToSTLFile( newValue );
  }
  else
  {
    GateVolumeMessenger::SetNewValue( command, newValue );
  }
}
