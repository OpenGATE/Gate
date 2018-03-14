/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateBoxMessenger.hh"
#include "GateBox.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//---------------------------------------------------------------------------
GateBoxMessenger::GateBoxMessenger(GateBox *itsCreator)
: GateVolumeMessenger(itsCreator)
{
  
//  G4cout << " DEBUT GateBoxMessenger\n";
  
  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName;
  cmdName = dir + "setXLength"; 
//  G4cout << " ########################"<< Gateendl;
//  G4cout << " ### GetDirectoryName() = " << cmdName << Gateendl;  
  pBoxXLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pBoxXLengthCmd->SetGuidance("Set length along X of the box.");
  pBoxXLengthCmd->SetParameterName("Length",false);
  pBoxXLengthCmd->SetRange("Length>0.");
  pBoxXLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setYLength";
//  G4cout << " ########################"<< Gateendl;
//  G4cout << " ### GetDirectoryName() " << cmdName << Gateendl;
  pBoxYLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pBoxYLengthCmd->SetGuidance("Set length along Y of the box.");
  pBoxYLengthCmd->SetParameterName("Length",false);
  pBoxYLengthCmd->SetRange("Length>0.");
  pBoxYLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
//  G4cout << " ########################"<< Gateendl;
//  G4cout << " ### GetDirectoryName() " << cmdName << Gateendl;
  pBoxZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pBoxZLengthCmd->SetGuidance("Set length along Z of the box.");
  pBoxZLengthCmd->SetParameterName("Length",false);
  pBoxZLengthCmd->SetRange("Length>0.");
  pBoxZLengthCmd->SetUnitCategory("Length");

//  G4cout << " FIN GateBoxMessenger\n";
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
GateBoxMessenger::~GateBoxMessenger()
{
  delete pBoxXLengthCmd;
  delete pBoxYLengthCmd;
  delete pBoxZLengthCmd;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateBoxMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{ 
  if( command==pBoxXLengthCmd )
    { 
//    G4cout << " ^^^ command==ppBoxXLengthCmd\n";
//    G4cout << " ^^^ ppBoxXLengthCmd->GetNewDoubleValue(newValue) = " << ppBoxXLengthCmd->GetNewDoubleValue(newValue) << Gateendl;
//    G4cout << " SetNewValue GateBoxMessenger XLength = " << newValue << Gateendl;
    
    GetBoxCreator()->SetBoxXLength(pBoxXLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  
  else if( command==pBoxYLengthCmd )
    { 
//     G4cout << " ^^^ command==BoxYLengthCmd\n";
//     G4cout << " ^^^ BoxYLengthCmd->GetNewDoubleValue(newValue) = " << BoxYLengthCmd->GetNewDoubleValue(newValue) << Gateendl;
    GetBoxCreator()->SetBoxYLength(pBoxYLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
    
  else if( command==pBoxZLengthCmd )
    { 
//    G4cout << " ^^^ command==BoxZLengthCmd\n";
//    G4cout << " ^^^ BoxZLengthCmd->GetNewDoubleValue(newValue) = " << BoxZLengthCmd->GetNewDoubleValue(newValue) << Gateendl;
    
    GetBoxCreator()->SetBoxZLength(pBoxZLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   

  
  else
//    G4cout << " ^^^ GateVolumeMessenger::SetNewValue\n";
    GateVolumeMessenger::SetNewValue(command,newValue);

}
//---------------------------------------------------------------------------
