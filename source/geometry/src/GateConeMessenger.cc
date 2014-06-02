/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateConeMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateCone.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateConeMessenger::GateConeMessenger(GateCone *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 

  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName;

  cmdName = dir+"setRmin1";
  ConeRmin1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ConeRmin1Cmd->SetGuidance("Set internal radius on one side of the cone (0 for full cone).");
  ConeRmin1Cmd->SetParameterName("Rmin1",false);
  ConeRmin1Cmd->SetRange("Rmin1>=0.");
  ConeRmin1Cmd->SetUnitCategory("Length");

  cmdName = dir+"setRmax1";
  ConeRmax1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ConeRmax1Cmd->SetGuidance("Set external radius on one side of the cone.");
  ConeRmax1Cmd->SetParameterName("Rmax1",false);
  ConeRmax1Cmd->SetRange("Rmax1>0.");
  ConeRmax1Cmd->SetUnitCategory("Length");

  cmdName = dir+"setRmin2";
  ConeRmin2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ConeRmin2Cmd->SetGuidance("Set internal radius on one side of the cone (0 for full cone).");
  ConeRmin2Cmd->SetParameterName("Rmin2",false);
  ConeRmin2Cmd->SetRange("Rmin2>=0.");
  ConeRmin2Cmd->SetUnitCategory("Length");

  cmdName = dir+"setRmax2";
  ConeRmax2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ConeRmax2Cmd->SetGuidance("Set external radius on one side of the cone.");
  ConeRmax2Cmd->SetParameterName("Rmax2",false);
  ConeRmax2Cmd->SetRange("Rmax2>0.");
  ConeRmax2Cmd->SetUnitCategory("Length");

  cmdName = dir+"setHeight";
  ConeHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  ConeHeightCmd->SetGuidance("Set height of the cone.");
  ConeHeightCmd->SetParameterName("Height",false);
  ConeHeightCmd->SetRange("Height>0.");
  ConeHeightCmd->SetUnitCategory("Length");

  cmdName = dir+"setPhiStart";
  ConeSPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ConeSPhiCmd->SetGuidance("Set start phi angle.");
  ConeSPhiCmd->SetParameterName("PhiStart",false);
  ConeSPhiCmd->SetUnitCategory("Angle");

  cmdName = dir+"setDeltaPhi";
  ConeDPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  ConeDPhiCmd->SetGuidance("Set phi angular span (2PI for full cone).");
  ConeDPhiCmd->SetParameterName("DeltaPhi",false);
  ConeDPhiCmd->SetUnitCategory("Angle");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateConeMessenger::~GateConeMessenger()
{
    delete ConeHeightCmd;
    delete ConeRmin1Cmd;
    delete ConeRmax1Cmd;
    delete ConeRmin2Cmd;
    delete ConeRmax2Cmd;
    delete ConeSPhiCmd;
    delete ConeDPhiCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateConeMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == ConeRmin1Cmd )
    { GetConeCreator()->SetConeRmin1(ConeRmin1Cmd->GetNewDoubleValue(newValue));}
  else if( command == ConeRmax1Cmd )
    { GetConeCreator()->SetConeRmax1(ConeRmax1Cmd->GetNewDoubleValue(newValue));}
  else if( command == ConeRmin2Cmd )
    { GetConeCreator()->SetConeRmin2(ConeRmin2Cmd->GetNewDoubleValue(newValue));}
  else if( command == ConeRmax2Cmd )
    { GetConeCreator()->SetConeRmax2(ConeRmax2Cmd->GetNewDoubleValue(newValue));}
  else if( command==ConeHeightCmd )
    { GetConeCreator()->SetConeHeight(ConeHeightCmd->GetNewDoubleValue(newValue));}   
  else if( command == ConeSPhiCmd )
    { GetConeCreator()->SetConeSPhi(ConeSPhiCmd->GetNewDoubleValue(newValue));}
  else if( command == ConeDPhiCmd )
    { GetConeCreator()->SetConeDPhi(ConeDPhiCmd->GetNewDoubleValue(newValue));}
  else
    GateVolumeMessenger::SetNewValue(command,newValue);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
