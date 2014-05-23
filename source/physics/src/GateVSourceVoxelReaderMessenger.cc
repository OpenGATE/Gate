/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


//      ------------ GateVSourceVoxelReaderMessenger  ------
//           by G.Santin (14 Nov 2001)
// ************************************************************


#include "GateVSourceVoxelReaderMessenger.hh"
#include "GateVSourceVoxelReader.hh"
#include "GateVSource.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVSourceVoxelReaderMessenger::GateVSourceVoxelReaderMessenger(GateVSourceVoxelReader* voxelReader)
  : GateMessenger(G4String("source/") + 
		  voxelReader->GetSource()->GetName() + 
		  G4String("/") +
		  voxelReader->GetName(), 
		  true),
  m_voxelReader(voxelReader)
{ 

  G4String cmdName;

  cmdName = GetDirectoryName()+"setPosition";
  PositionCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  PositionCmd->SetGuidance("Set source position");
  PositionCmd->SetGuidance("1. 3-vector of source position");
  PositionCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setVoxelSize";
  VoxelSizeCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  VoxelSizeCmd->SetGuidance("Set source voxel size");
  PositionCmd->SetGuidance("1. 3-vector of voxel size");
  VoxelSizeCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"translator/insert";
  InsertTranslatorCmd = new GateUIcmdWithAVector<G4String>(cmdName,this);
  InsertTranslatorCmd->SetGuidance("Insert a translator");
  InsertTranslatorCmd->SetGuidance("1. Translator type");

  cmdName = GetDirectoryName()+"translator/remove";
  RemoveTranslatorCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RemoveTranslatorCmd->SetGuidance("Remove the translator");

  cmdName = GetDirectoryName()+"verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set GATE source voxel reader verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");
  
      cmdName = GetDirectoryName()+"SetTimeActivityTablesFrom";
  TimeActivTablesCmd = new G4UIcmdWithAString(cmdName,this);

  cmdName = GetDirectoryName()+"SetTimeSampling";
  SetTimeSamplingCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);

  
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVSourceVoxelReaderMessenger::~GateVSourceVoxelReaderMessenger()
{
  delete InsertTranslatorCmd;
  delete RemoveTranslatorCmd;
  delete VoxelSizeCmd;
  delete PositionCmd;
  delete VerboseCmd;
  delete TimeActivTablesCmd;
  delete SetTimeSamplingCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateVSourceVoxelReaderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ if( command == SetTimeSamplingCmd ) {
    m_voxelReader->SetTimeSampling( SetTimeSamplingCmd->GetNewDoubleValue( newValue ) );
    return;}
  else if( command == TimeActivTablesCmd ) {
    m_voxelReader->SetTimeActivTables( newValue );
    return;}
  else if( command == PositionCmd ) {
    m_voxelReader->SetPosition(PositionCmd->GetNew3VectorValue(newValue));
  } else if( command == VoxelSizeCmd) {
    m_voxelReader->SetVoxelSize(VoxelSizeCmd->GetNew3VectorValue(newValue));
  } else if( command == InsertTranslatorCmd) {
    m_voxelReader->InsertTranslator(InsertTranslatorCmd->GetNewVectorValue(newValue)[0]);
  } else if( command == RemoveTranslatorCmd) {
    m_voxelReader->RemoveTranslator();
  } else if( command == VerboseCmd) {
    m_voxelReader->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



