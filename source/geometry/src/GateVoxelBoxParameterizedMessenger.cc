/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVoxelBoxParameterizedMessenger.hh"
#include "GateVoxelBoxParameterized.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"




//-------------------------------------------------------------------------------------------
GateVoxelBoxParameterizedMessenger::GateVoxelBoxParameterizedMessenger(GateVoxelBoxParameterized *itsInserter)
  :GateMessenger(itsInserter->GetObjectName()+"/geometry"),
   m_inserter(itsInserter)
{ 
 
  // G4cout << "GateVoxelBoxParameterizedMessenger::GateVoxelBoxParameterizedMessenger - Entered " << GetDirectoryName() << G4endl; 

  GetDirectory()->SetGuidance("Control the parameterized geometry.");

  G4String cmdName;
  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/attachVoxelPhantomSD"; // RTA
  AttachPhantomSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  AttachPhantomSDCmd->SetGuidance("Attach the phantom-SD to the matrix.");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/addOutput"; 
  AddOutputCmd = new G4UIcmdWithAString(cmdName,this);
  AddOutputCmd->SetGuidance("adds an output module to write the dose matrix");
  AddOutputCmd->SetGuidance("1. output module name");

  cmdName = GetDirectoryName()+"insertReader";
  InsertReaderCmd = new G4UIcmdWithAString(cmdName,this);
  InsertReaderCmd->SetGuidance("Insert a reader of the type specified");
  InsertReaderCmd->SetGuidance("1. reader type");

  cmdName = GetDirectoryName()+"removeReader";
  RemoveReaderCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RemoveReaderCmd->SetGuidance("Remove the reader");

}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
GateVoxelBoxParameterizedMessenger::~GateVoxelBoxParameterizedMessenger()
{
   delete InsertReaderCmd;
   delete RemoveReaderCmd;
   delete AttachPhantomSDCmd;
   delete AddOutputCmd;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
void GateVoxelBoxParameterizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==AttachPhantomSDCmd )
    {       
      GetVoxelParameterizedInserter()->AttachPhantomSD();  }   

  else if ( command == AddOutputCmd )
    { GetVoxelParameterizedInserter()->AddOutput(newValue); }

  else if ( command == InsertReaderCmd )
    { G4cout << " Call VoxelParameterized::InsertReader"  << G4endl; GetVoxelParameterizedInserter()->InsertReader(newValue); }

  else if ( command == RemoveReaderCmd )
    { GetVoxelParameterizedInserter()->RemoveReader(); }

  else
    GateMessenger::SetNewValue(command,newValue);

}
//-------------------------------------------------------------------------------------------
