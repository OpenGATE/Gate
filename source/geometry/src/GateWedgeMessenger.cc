/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateWedgeMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateWedge.hh"

class GateVolumeMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateWedgeMessenger::GateWedgeMessenger(GateWedge *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 
  G4String dir = GetDirectoryName() + "geometry/";
  G4String cmdName = dir+"setXLength";
  WedgeXLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  WedgeXLengthCmd->SetGuidance("Set length along X of the Wedge.");
  WedgeXLengthCmd->SetParameterName("Length",false);
  WedgeXLengthCmd->SetRange("Length>0.");
  WedgeXLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setNarrowerXLength";
  WedgeNarrowerXLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  WedgeNarrowerXLengthCmd->SetGuidance("Set length along X at the narrower side of the Wedge.");
  WedgeNarrowerXLengthCmd->SetParameterName("Length",false);
  WedgeNarrowerXLengthCmd->SetRange("Length>0.");
  WedgeNarrowerXLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setYLength";
  WedgeYLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  WedgeYLengthCmd->SetGuidance("Set length along Y of the Wedge.");
  WedgeYLengthCmd->SetParameterName("Length",false);
  WedgeYLengthCmd->SetRange("Length>0.");
  WedgeYLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setZLength";
  WedgeZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  WedgeZLengthCmd->SetGuidance("Set length along Z of the Wedge.");
  WedgeZLengthCmd->SetParameterName("Length",false);
  WedgeZLengthCmd->SetRange("Length>0.");
  WedgeZLengthCmd->SetUnitCategory("Length");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateWedgeMessenger::~GateWedgeMessenger()
{
  delete WedgeXLengthCmd;
  delete WedgeNarrowerXLengthCmd;
  delete WedgeYLengthCmd;
  delete WedgeZLengthCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateWedgeMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==WedgeXLengthCmd )
    { GetWedgeCreator()->SetWedgeXLength(WedgeXLengthCmd->GetNewDoubleValue(newValue));}   

  else if( command==WedgeNarrowerXLengthCmd )
    { GetWedgeCreator()->SetWedgeNarrowerXLength(WedgeNarrowerXLengthCmd->GetNewDoubleValue(newValue));}   
  
  else if( command==WedgeYLengthCmd )
    { GetWedgeCreator()->SetWedgeYLength(WedgeYLengthCmd->GetNewDoubleValue(newValue));}   
  
  else if( command==WedgeZLengthCmd )
    { GetWedgeCreator()->SetWedgeZLength(WedgeZLengthCmd->GetNewDoubleValue(newValue));}   
  
  else
    GateVolumeMessenger::SetNewValue(command,newValue);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
