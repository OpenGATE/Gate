/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateCCHitFileReaderMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateCCHitFileReader.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateCCHitFileReaderMessenger::GateCCHitFileReaderMessenger(GateCCHitFileReader* itsHitFileReader)
  : GateClockDependentMessenger(itsHitFileReader)
{
  SetDirectoryGuidance("Control the parameters of the hit-file reader used for DigiGate");

  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the input ROOT hit data file");
  SetFileNameCmd->SetParameterName("Name",false);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateCCHitFileReaderMessenger::~GateCCHitFileReaderMessenger()
{
  delete SetFileNameCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateCCHitFileReaderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == SetFileNameCmd)
    GetCCHitFileReader()->SetFileName(newValue);
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
