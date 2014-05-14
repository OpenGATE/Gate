/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateHitFileReaderMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateHitFileReader.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateHitFileReaderMessenger::GateHitFileReaderMessenger(GateHitFileReader* itsHitFileReader)
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

GateHitFileReaderMessenger::~GateHitFileReaderMessenger()
{
  delete SetFileNameCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateHitFileReaderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == SetFileNameCmd)
    GetHitFileReader()->SetFileName(newValue);
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
