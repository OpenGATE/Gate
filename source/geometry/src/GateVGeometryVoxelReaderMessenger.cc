/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVGeometryVoxelReaderMessenger.hh"
#include "GateVGeometryVoxelReader.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//----------------------------------------------------------------------------------------------------------
GateVGeometryVoxelReaderMessenger::GateVGeometryVoxelReaderMessenger(GateVGeometryVoxelReader* voxelReader)
  : GateVGeometryVoxelStoreMessenger(voxelReader)
    , m_voxelReader(voxelReader)
{ 

  G4String cmdName;

  cmdName = GetDirectoryName()+"readFile";
  ReadFileCmd = new G4UIcmdWithAString(cmdName,this);
  ReadFileCmd->SetGuidance("Reads the image from a file");
  ReadFileCmd->SetGuidance("1. file name");

  cmdName = GetDirectoryName()+"describe";
  DescribeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  DescribeCmd->SetGuidance("Description of the reader status");
  DescribeCmd->SetGuidance("1. verbosity level");

  cmdName = GetDirectoryName()+"insertTranslator";
  InsertTranslatorCmd = new G4UIcmdWithAString(cmdName,this);
  InsertTranslatorCmd->SetGuidance("Insert a translator of the type specified");
  InsertTranslatorCmd->SetGuidance("1. translator type");

  cmdName = GetDirectoryName()+"removeTranslator";
  RemoveTranslatorCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RemoveTranslatorCmd->SetGuidance("Remove the translator");

}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
GateVGeometryVoxelReaderMessenger::~GateVGeometryVoxelReaderMessenger()
{
   delete InsertTranslatorCmd;
   delete RemoveTranslatorCmd;
   delete ReadFileCmd;
   delete DescribeCmd;
}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
void GateVGeometryVoxelReaderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ReadFileCmd ) {
    m_voxelReader->ReadFile(newValue);
  } else if ( command == DescribeCmd ) {
    m_voxelReader->Describe(DescribeCmd->GetNewIntValue(newValue));
  } else if ( command == InsertTranslatorCmd ) {
    m_voxelReader->InsertTranslator(newValue);
  } else if ( command == RemoveTranslatorCmd ) {
    m_voxelReader->RemoveTranslator();
  } else {
    GateVGeometryVoxelStoreMessenger::SetNewValue(command, newValue);
  }
}
//----------------------------------------------------------------------------------------------------------



