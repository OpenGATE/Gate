/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSphereRepeaterMessenger.hh"
#include "GateSphereRepeater.hh"


#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateSphereRepeaterMessenger::GateSphereRepeaterMessenger(GateSphereRepeater* itsSphereRepeater)
  :GateObjectRepeaterMessenger(itsSphereRepeater)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setRepeatNumberWithTheta";
    SetRepeatNumberWithThetaCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetRepeatNumberWithThetaCmd->SetGuidance("Set the number of copies of the object along the ring.");
    SetRepeatNumberWithThetaCmd->SetParameterName("Ntheta",false);
    SetRepeatNumberWithThetaCmd->SetRange("Ntheta >= 1");

    cmdName = GetDirectoryName()+"setRepeatNumberWithPhi";
    SetRepeatNumberWithPhiCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetRepeatNumberWithPhiCmd->SetGuidance("Set the number of copies of the object along the normal direction to the ring.");
    SetRepeatNumberWithPhiCmd->SetParameterName("Nphi",false);
    SetRepeatNumberWithPhiCmd->SetRange("Nphi >= 1");

    cmdName = GetDirectoryName()+"autoCenter";
    AutoCenterCmd = new G4UIcmdWithABool(cmdName,this);
    AutoCenterCmd->SetGuidance("Enable or disable auto-centering.");
    AutoCenterCmd->SetParameterName("flag",true);
    AutoCenterCmd->SetDefaultValue(true);

    cmdName = GetDirectoryName()+"enableAutoRotation";
    EnableAutoRotationCmd = new G4UIcmdWithABool(cmdName,this);
    EnableAutoRotationCmd->SetGuidance("Enable the object auto-rotation option.");
    EnableAutoRotationCmd->SetParameterName("flag",true);
    EnableAutoRotationCmd->SetDefaultValue(true);

    cmdName = GetDirectoryName()+"disableAutoRotation";
    DisableAutoRotationCmd = new G4UIcmdWithABool(cmdName,this);
    DisableAutoRotationCmd->SetGuidance("Disable the object auto-rotation option.");
    DisableAutoRotationCmd->SetParameterName("flag",true);
    DisableAutoRotationCmd->SetDefaultValue(true);

    cmdName = GetDirectoryName()+"setThetaAngle";
    ThetaAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    ThetaAngleCmd->SetGuidance("Set the rotation angle difference between two copies of the replicated volume along the ring.");
    ThetaAngleCmd->SetParameterName("Dtheta",false);
    ThetaAngleCmd->SetUnitCategory("Angle");

    cmdName = GetDirectoryName()+"setPhiAngle";
    PhiAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    PhiAngleCmd->SetGuidance("Set the rotation angle difference between two copies of the replicated volume along the normal direction to the ring.");
    PhiAngleCmd->SetParameterName("Dphi",false);
    PhiAngleCmd->SetUnitCategory("Angle");

    
    cmdName = GetDirectoryName()+"setRadius";
    RadiusCmd  = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    RadiusCmd->SetGuidance("Set the radius of the sphere along which the volume will be replicated");
    RadiusCmd->SetParameterName("radius",false);
    RadiusCmd->SetUnitCategory("Length");
 }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateSphereRepeaterMessenger::~GateSphereRepeaterMessenger()
{
    delete AutoCenterCmd;
    delete EnableAutoRotationCmd;
    delete DisableAutoRotationCmd;
    delete SetRepeatNumberWithPhiCmd;
    delete SetRepeatNumberWithThetaCmd;
    delete ThetaAngleCmd;
    delete PhiAngleCmd;
    delete RadiusCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateSphereRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==SetRepeatNumberWithPhiCmd )
    { GetSphereRepeater()->SetRepeatNumberWithPhi(SetRepeatNumberWithPhiCmd->GetNewIntValue(newValue));}   
  else if( command==SetRepeatNumberWithThetaCmd )
    { GetSphereRepeater()->SetRepeatNumberWithTheta(SetRepeatNumberWithThetaCmd->GetNewIntValue(newValue));}   
  else if( command==AutoCenterCmd )
    { GetSphereRepeater()->SetAutoCenterFlag(AutoCenterCmd->GetNewBoolValue(newValue));} 
  else if ( command==EnableAutoRotationCmd )
     { GetSphereRepeater()->SetAutoRotation(EnableAutoRotationCmd->GetNewBoolValue(newValue));}   
  else if ( command==DisableAutoRotationCmd )
     { GetSphereRepeater()->SetAutoRotation(!(DisableAutoRotationCmd->GetNewBoolValue(newValue)));} 
  else if( command==ThetaAngleCmd )
    { GetSphereRepeater()->SetThetaAngle(ThetaAngleCmd->GetNewDoubleValue(newValue)) ;}
  else if( command==PhiAngleCmd )
    { GetSphereRepeater()->SetPhiAngle(PhiAngleCmd->GetNewDoubleValue(newValue)) ;}
  else if( command==RadiusCmd )
    { GetSphereRepeater()->SetRadius(RadiusCmd->GetNewDoubleValue(newValue)) ;}
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
