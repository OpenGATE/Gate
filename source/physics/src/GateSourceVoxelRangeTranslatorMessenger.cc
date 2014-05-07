/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


//      ------------ GateSourceVoxelRangeTranslatorMessenger  ------
//           by G.Santin (14 Nov 2001)
// ************************************************************


#include "GateSourceVoxelRangeTranslatorMessenger.hh"
#include "GateSourceVoxelRangeTranslator.hh"
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

GateSourceVoxelRangeTranslatorMessenger::GateSourceVoxelRangeTranslatorMessenger(GateSourceVoxelRangeTranslator* voxelTranslator)
  : GateMessenger(G4String("source/") + 
		  voxelTranslator->GetReader()->GetSource()->GetName() + 
		  G4String("/") +
		  voxelTranslator->GetReader()->GetName() + 
		  G4String("/") +
		  voxelTranslator->GetName(), 
		  true),
  m_voxelTranslator(voxelTranslator)
{ 

  G4String cmdName;

  cmdName = GetDirectoryName()+"readTable";
  ReadTableCmd = new G4UIcmdWithAString(cmdName,this);
  ReadTableCmd->SetGuidance("Reads the translation table from a file");
  ReadTableCmd->SetGuidance("1. file name");

  cmdName = GetDirectoryName()+"describe";
  DescribeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  DescribeCmd->SetGuidance("Description of the translator status");
  DescribeCmd->SetGuidance("1. verbosity level");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateSourceVoxelRangeTranslatorMessenger::~GateSourceVoxelRangeTranslatorMessenger()
{
   delete ReadTableCmd;
   delete DescribeCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateSourceVoxelRangeTranslatorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ReadTableCmd ) {
    m_voxelTranslator->ReadTranslationTable(newValue);
  } else if ( command == DescribeCmd ) {
    m_voxelTranslator->Describe(DescribeCmd->GetNewIntValue(newValue));
  } 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



