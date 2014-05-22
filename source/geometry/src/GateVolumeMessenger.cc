/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateVolumeMessenger.hh"
#include "GateVVolume.hh"
#include "GateVisAttributesMessenger.hh"

//-------------------------------------------------------------------------------------
GateVolumeMessenger::GateVolumeMessenger(GateVVolume* itsCreator, const G4String& itsDirectoryName)
  //: GateClockDependentMessenger(itsCreator, itsCreator->GetObjectName()+"/geometry"),
  : GateClockDependentMessenger(itsCreator, itsDirectoryName)
{ 
    
  G4String guidance = G4String("Controls the geometry for '") + GetVolumeCreator()->GetObjectName() +"'";

  GetDirectory()->SetGuidance(guidance.c_str());

  //  G4cout << " before new GateVisAttributesMessenger creating GetVolumeCreator() ="  << GetVolumeCreator()->GetObjectName() << G4endl;  
  pVisAttributesMessenger  = 
      	new GateVisAttributesMessenger(GetVolumeCreator()->GetVisAttributes(), GetVolumeCreator()->GetObjectName()+"/vis");
  
  G4String cmdName;
  cmdName = GetDirectoryName()+"setMaterial"; 
  
  pSetMaterCmd = new G4UIcmdWithAString(cmdName.c_str(),this);
  pSetMaterCmd->SetGuidance("Select Material of the detector");
  pSetMaterCmd->SetParameterName("choice",false);
  //  pSetMaterCmd->AvailableForStates(G4State_Idle);

  cmdName = GetDirectoryName()+"attachCrystalSD";
  pAttachCrystalSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  pAttachCrystalSDCmd->SetGuidance("Attach the crystal-SD to the object.");

  cmdName = GetDirectoryName()+"attachPhantomSD";
  pAttachPhantomSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  pAttachPhantomSDCmd->SetGuidance("Attach the phantom-SD to the object.");

}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
GateVolumeMessenger::~GateVolumeMessenger()
{
   delete pVisAttributesMessenger;
   delete pSetMaterCmd;
   delete pAttachCrystalSDCmd;
   delete pAttachPhantomSDCmd;
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
void GateVolumeMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{     
    if( command == pSetMaterCmd )
   { 
    GetVolumeCreator()->SetMaterialName(newValue);
   }
   else if( command==pAttachCrystalSDCmd )
    { 
      GetVolumeCreator()->AttachCrystalSD();
    } 
   else if( command==pAttachPhantomSDCmd )
    { 
      GetVolumeCreator()->AttachPhantomSD();
    }  
   else{
    GateClockDependentMessenger::SetNewValue(command,newValue);
   }
}
//-------------------------------------------------------------------------------------
