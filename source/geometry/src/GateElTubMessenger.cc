/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateElTubMessenger.hh"
#include "GateElTub.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"



//-------------------------------------------------------------------------------------------------------------------------
GateElTubMessenger::GateElTubMessenger(GateElTub *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 

  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName = dir +"setShort";
  pElTubRshortCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pElTubRshortCmd->SetGuidance("Set short radius of the Elliptical Tub.");
  pElTubRshortCmd->SetParameterName("Rshort",false);
  pElTubRshortCmd->SetRange("Rshort>=0.");
  pElTubRshortCmd->SetUnitCategory("Length");

  cmdName = dir+"setLong";
  pElTubRlongCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  pElTubRlongCmd->SetGuidance("Set long radius of the Elliptical Tub.");
  pElTubRlongCmd->SetParameterName("Rlong",false);
  pElTubRlongCmd->SetRange("Rlong>0.");
  pElTubRlongCmd->SetUnitCategory("Length");

  cmdName = dir+"setHeight";
  pElTubHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  pElTubHeightCmd->SetGuidance("Set height of the Elliptical Tub.");
  pElTubHeightCmd->SetParameterName("Height",false);
  pElTubHeightCmd->SetRange("Height>0.");
  pElTubHeightCmd->SetUnitCategory("Length");

}
//-------------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------------
GateElTubMessenger::~GateElTubMessenger()
{
    delete pElTubHeightCmd;
    delete pElTubRshortCmd;
    delete pElTubRlongCmd;
}
//-------------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------------
void GateElTubMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == pElTubRshortCmd )
    { 
     GetElTubCreator()->SetElTubRshort(pElTubRshortCmd->GetNewDoubleValue(newValue));/*TellGeometryToUpdate();*/}
  else if( command == pElTubRlongCmd )
    { GetElTubCreator()->SetElTubRlong(pElTubRlongCmd->GetNewDoubleValue(newValue));/*TellGeometryToUpdate();*/}
  else if( command==pElTubHeightCmd )
    { GetElTubCreator()->SetElTubHeight(pElTubHeightCmd->GetNewDoubleValue(newValue)); /*TellGeometryToUpdate();*/}   
  else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------------------------------------------
