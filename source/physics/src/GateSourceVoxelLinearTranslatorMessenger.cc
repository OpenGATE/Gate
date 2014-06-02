/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

//      ------------ GateSourceVoxelLinearTranslatorMessenger  ------
//           by G.Santin (14 Nov 2001)
// ************************************************************


#include "GateSourceVoxelLinearTranslatorMessenger.hh"
#include "GateSourceVoxelLinearTranslator.hh"
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

GateSourceVoxelLinearTranslatorMessenger::GateSourceVoxelLinearTranslatorMessenger(GateSourceVoxelLinearTranslator* voxelTranslator)
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

  cmdName = GetDirectoryName()+"setScale";
  ValueToActivityScaleCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  ValueToActivityScaleCmd->SetGuidance("Set scale between input value and source activity");
  ValueToActivityScaleCmd->SetGuidance("1. scale value and unit");
  ValueToActivityScaleCmd->SetUnitCategory("Activity");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateSourceVoxelLinearTranslatorMessenger::~GateSourceVoxelLinearTranslatorMessenger()
{
   delete ValueToActivityScaleCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateSourceVoxelLinearTranslatorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ValueToActivityScaleCmd ) {
    m_voxelTranslator->SetValueToActivityScale(ValueToActivityScaleCmd->GetNewDoubleValue(newValue));
  } 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



