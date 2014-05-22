/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSphereMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateSphere.hh"

//-------------------------------------------------------------------------------------
GateSphereMessenger::GateSphereMessenger(GateSphere *itsCreator)
: GateVolumeMessenger(itsCreator)
{ 

  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName = dir+"setRmin";
  SphereRminCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  
  G4cout << " Rmin" << cmdName.c_str() << G4endl;
  
  SphereRminCmd->SetGuidance("Set internal radius of the sphere (0 for full sphere).");
  SphereRminCmd->SetParameterName("Rmin",false);
  SphereRminCmd->SetRange("Rmin>=0.");
  SphereRminCmd->SetUnitCategory("Length");

  cmdName = dir+"setRmax";
  SphereRmaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  SphereRmaxCmd->SetGuidance("Set external radius of the sphere.");
  SphereRmaxCmd->SetParameterName("Rmax",false);
  SphereRmaxCmd->SetRange("Rmax>0.");
  SphereRmaxCmd->SetUnitCategory("Length");

  cmdName = dir+"setPhiStart";
  SphereSPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  SphereSPhiCmd->SetGuidance("Set start phi angle.");
  SphereSPhiCmd->SetParameterName("PhiStart",false);
  SphereSPhiCmd->SetUnitCategory("Angle");

  cmdName = dir+"setDeltaPhi";
  SphereDPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  SphereDPhiCmd->SetGuidance("Set phi angular span (2PI for full sphere).");
  SphereDPhiCmd->SetParameterName("DeltaPhi",false);
  SphereDPhiCmd->SetUnitCategory("Angle");

  cmdName = dir+"setThetaStart";
  SphereSThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  SphereSThetaCmd->SetGuidance("Set start theta angle.");
  SphereSThetaCmd->SetParameterName("ThetaStart",false);
  SphereSThetaCmd->SetUnitCategory("Angle");

  cmdName = dir+"setDeltaTheta";
  SphereDThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  SphereDThetaCmd->SetGuidance("Set theta angular span (2PI for full sphere).");
  SphereDThetaCmd->SetParameterName("DeltaTheta",false);
  SphereDThetaCmd->SetUnitCategory("Angle");

}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
GateSphereMessenger::~GateSphereMessenger()
{
    delete SphereRminCmd;
    delete SphereRmaxCmd;
    delete SphereSPhiCmd;
    delete SphereDPhiCmd;
    delete SphereSThetaCmd;
    delete SphereDThetaCmd;
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateSphereMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{ 
  if( command == SphereRminCmd )
    { 
    GetSphereCreator()->SetSphereRmin(SphereRminCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}
  else if( command == SphereRmaxCmd )
    { GetSphereCreator()->SetSphereRmax(SphereRmaxCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}
  else if( command == SphereSPhiCmd )
    { GetSphereCreator()->SetSphereSPhi(SphereSPhiCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}
  else if( command == SphereDPhiCmd )
    { GetSphereCreator()->SetSphereDPhi(SphereDPhiCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}
  else if( command == SphereSThetaCmd )
    { GetSphereCreator()->SetSphereSTheta(SphereSThetaCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}
  else if( command == SphereDThetaCmd )
    { GetSphereCreator()->SetSphereDTheta(SphereDThetaCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}
  else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------
