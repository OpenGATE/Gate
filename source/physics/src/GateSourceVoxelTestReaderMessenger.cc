/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceVoxelTestReaderMessenger.hh"
#include "GateSourceVoxelTestReader.hh"

GateSourceVoxelTestReaderMessenger::GateSourceVoxelTestReaderMessenger(GateSourceVoxelTestReader* voxelReader)
  : GateVSourceVoxelReaderMessenger(voxelReader)
{
  G4String cmdName;
  cmdName = GetDirectoryName()+"readFile";
  ReadFileCmd = new GateUIcmdWithAVector<G4String>(cmdName,this);
  ReadFileCmd->SetGuidance("Read a file with the voxel activities");
  ReadFileCmd->SetGuidance("1. File name");
}

GateSourceVoxelTestReaderMessenger::~GateSourceVoxelTestReaderMessenger()
{
   delete ReadFileCmd;
}

void GateSourceVoxelTestReaderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ReadFileCmd ) {
    m_voxelReader->ReadFile(ReadFileCmd->GetNewVectorValue(newValue)[0]);
  } else {
    GateVSourceVoxelReaderMessenger::SetNewValue(command, newValue);
  }
}
