/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \file GateVoxelOutputMessenger.cc
 */
#include "GateVoxelOutputMessenger.hh"
#include "GateVoxelOutput.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVoxelOutputMessenger::GateVoxelOutputMessenger(GateVoxelOutput* g)
  : GateOutputModuleMessenger(g)
  , m_gateVoxelOutput(g)
{ 
  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Sets the name of the Dose Matrix output file");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"saveUncertainty";
  saveUncertaintyCmd = new G4UIcmdWithABool(cmdName,this);
  saveUncertaintyCmd->SetGuidance("Determines (true|false) if the square of the deposits is to be saved");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVoxelOutputMessenger::~GateVoxelOutputMessenger()
{
  delete SetFileNameCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateVoxelOutputMessenger::SetNewValue(G4UIcommand* command,G4String newValue){

  if (command == SetFileNameCmd){
    m_gateVoxelOutput->SetFileName(newValue);
  }
  else if(command == saveUncertaintyCmd){
        m_gateVoxelOutput->SetSaveUncertainty(saveUncertaintyCmd->GetNewBoolValue(newValue) );
  }
  else {
    GateOutputModuleMessenger::SetNewValue(command,newValue);
  }

}

