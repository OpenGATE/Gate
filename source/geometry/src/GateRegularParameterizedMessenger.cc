/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateRegularParameterizedMessenger.hh"
#include "GateRegularParameterized.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//-------------------------------------------------------------------------------------------
///////////////////
//  Constructor  //
///////////////////

GateRegularParameterizedMessenger::GateRegularParameterizedMessenger(GateRegularParameterized *itsInserter)
  :GateMessenger(itsInserter->GetObjectName()+"/geometry"),m_inserter(itsInserter)
{
  GetDirectory()->SetGuidance("Control the parameterized geometry.");
  G4String cmdName;

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/attachVoxelPhantomSD";
  AttachPhantomSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  AttachPhantomSDCmd->SetGuidance("Attach the phantom-SD to the matrix.");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/addOutput";
  AddOutputCmd = new G4UIcmdWithAString(cmdName,this);
  AddOutputCmd->SetGuidance("Adds an output module to write the dose matrix");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+"/setSkipEqualMaterials";
  SkipEqualMaterialsCmd = new G4UIcmdWithABool(cmdName,this);
  SkipEqualMaterialsCmd->SetGuidance("Skip or not boundaries when neighbour voxels are made of same material (default: yes)");

  cmdName = G4String("/gate/") + itsInserter->GetObjectName()+ "/verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set the verbosity level of the construction of the regular phantom, from 1 to 5 (default is 0)");

  cmdName = GetDirectoryName()+"insertReader";
  InsertReaderCmd = new G4UIcmdWithAString(cmdName,this);
  InsertReaderCmd->SetGuidance("Insert a reader of the type specified");

  cmdName = GetDirectoryName()+"removeReader";
  RemoveReaderCmd = new G4UIcmdWithoutParameter(cmdName,this);
  RemoveReaderCmd->SetGuidance("Remove the reader");
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
//////////////////
//  Destructor  //
//////////////////

GateRegularParameterizedMessenger::~GateRegularParameterizedMessenger()
{
   delete InsertReaderCmd;
   delete RemoveReaderCmd;
   delete AttachPhantomSDCmd;
   delete AddOutputCmd;
   delete SkipEqualMaterialsCmd;
   delete VerboseCmd;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
///////////////////
//  SetNewValue  //
///////////////////

void GateRegularParameterizedMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command==AttachPhantomSDCmd )
    { GetRegularParameterizedInserter()->AttachPhantomSD(); }

  else if ( command == AddOutputCmd )
    { GetRegularParameterizedInserter()->AddOutput(newValue); }

  else if ( command == SkipEqualMaterialsCmd )
    { GetRegularParameterizedInserter()->ChangeSkipEqualMaterials(SkipEqualMaterialsCmd->GetNewBoolValue(newValue)); }

  else if ( command == VerboseCmd )
    { GetRegularParameterizedInserter()->SetVerbosity(VerboseCmd->GetNewIntValue(newValue)); }

  else if ( command == InsertReaderCmd )
    { GetRegularParameterizedInserter()->InsertReader(newValue); }

  else if ( command == RemoveReaderCmd )
    { GetRegularParameterizedInserter()->RemoveReader(); }

  else
    GateMessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------------
