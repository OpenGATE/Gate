/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateGeometryVoxelTabulatedTranslatorMessenger.hh"
#include "GateGeometryVoxelTabulatedTranslator.hh"

#include "GateClock.hh"
//e#include "GateVVolume.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//-----------------------------------------------------------------------------------------------------------
GateGeometryVoxelTabulatedTranslatorMessenger::GateGeometryVoxelTabulatedTranslatorMessenger(GateGeometryVoxelTabulatedTranslator* voxelTranslator)
  : GateMessenger(voxelTranslator->GetReader()->GetCreator()->GetObjectName() + 
		  G4String("/") +
		  voxelTranslator->GetReader()->GetName() + 
		  G4String("/") +
		  voxelTranslator->GetName(), 
		  true),
  m_voxelTranslator(voxelTranslator)
{ 

  G4String cmdName;

  cmdName = GetDirectoryName()+"readTable";
  G4cout << " cmdName readTable = " << cmdName << G4endl;
  ReadTableCmd = new G4UIcmdWithAString(cmdName,this);
  ReadTableCmd->SetGuidance("Reads the translation table from a file");
  ReadTableCmd->SetGuidance("1. file name");

  cmdName = GetDirectoryName()+"describe";
  DescribeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  DescribeCmd->SetGuidance("Description of the translator status");
  DescribeCmd->SetGuidance("1. verbosity level");

}
//-----------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------
GateGeometryVoxelTabulatedTranslatorMessenger::~GateGeometryVoxelTabulatedTranslatorMessenger()
{
   delete ReadTableCmd;
   delete DescribeCmd;
}
//-----------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------
void GateGeometryVoxelTabulatedTranslatorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ReadTableCmd ) {
    G4cout << " Call ReadTableCmd command" << G4endl;
    m_voxelTranslator->ReadTranslationTable(newValue);
  } else if ( command == DescribeCmd ) {
    m_voxelTranslator->Describe(DescribeCmd->GetNewIntValue(newValue));
  } 
}
//-----------------------------------------------------------------------------------------------------------


