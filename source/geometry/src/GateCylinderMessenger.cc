/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCylinderMessenger.hh"
#include "GateCylinder.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"



//-------------------------------------------------------------------------------------------------------------------------
GateCylinderMessenger::GateCylinderMessenger(GateCylinder *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 

  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName = dir +"setRmin";
  pCylinderRminCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pCylinderRminCmd->SetGuidance("Set internal radius of the cylinder (0 for full cylinder).");
  pCylinderRminCmd->SetParameterName("Rmin",false);
  pCylinderRminCmd->SetRange("Rmin>=0.");
  pCylinderRminCmd->SetUnitCategory("Length");

  cmdName = dir+"setRmax";
  pCylinderRmaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pCylinderRmaxCmd->SetGuidance("Set external radius of the cylinder.");
  pCylinderRmaxCmd->SetParameterName("Rmax",false);
  pCylinderRmaxCmd->SetRange("Rmax>0.");
  pCylinderRmaxCmd->SetUnitCategory("Length");

  cmdName = dir+"setHeight";
  pCylinderHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  pCylinderHeightCmd->SetGuidance("Set height of the cylinder.");
  pCylinderHeightCmd->SetParameterName("Height",false);
  pCylinderHeightCmd->SetRange("Height>0.");
  pCylinderHeightCmd->SetUnitCategory("Length");

  cmdName = dir+"setPhiStart";
  pCylinderSPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pCylinderSPhiCmd->SetGuidance("Set start phi angle.");
  pCylinderSPhiCmd->SetParameterName("PhiStart",false);
  pCylinderSPhiCmd->SetUnitCategory("Angle");

  cmdName = dir +"setDeltaPhi";
  pCylinderDPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pCylinderDPhiCmd->SetGuidance("Set phi angular span (2PI for full cylinder).");
  pCylinderDPhiCmd->SetParameterName("DeltaPhi",false);
  pCylinderDPhiCmd->SetUnitCategory("Angle");

}
//-------------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------------
GateCylinderMessenger::~GateCylinderMessenger()
{
    delete pCylinderHeightCmd;
    delete pCylinderRminCmd;
    delete pCylinderRmaxCmd;
    delete pCylinderSPhiCmd;
    delete pCylinderDPhiCmd;
}
//-------------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------------
void GateCylinderMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == pCylinderRminCmd )
    { 
     GetCylinderCreator()->SetCylinderRmin(pCylinderRminCmd->GetNewDoubleValue(newValue));/*TellGeometryToUpdate();*/}
  else if( command == pCylinderRmaxCmd )
    { GetCylinderCreator()->SetCylinderRmax(pCylinderRmaxCmd->GetNewDoubleValue(newValue));/*TellGeometryToUpdate();*/}
  else if( command==pCylinderHeightCmd )
    { GetCylinderCreator()->SetCylinderHeight(pCylinderHeightCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  else if( command == pCylinderSPhiCmd )
    { GetCylinderCreator()->SetCylinderSPhi(pCylinderSPhiCmd->GetNewDoubleValue(newValue));/*TellGeometryToUpdate();*/}
  else if( command == pCylinderDPhiCmd )
    { GetCylinderCreator()->SetCylinderDPhi(pCylinderDPhiCmd->GetNewDoubleValue(newValue));/*TellGeometryToUpdate();*/}
  else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------------------------------------------
