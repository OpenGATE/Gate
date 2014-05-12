/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateClockDependentMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateClockDependent.hh"
#include "GateSPECTHeadSystem.hh"
#include "GateObjectStore.hh"
#include "GateVVolume.hh"

//----------------------------------------------------------------------------------------------------------
// Constructor
// The flags are passed to the base-class GateNamedObjectMessenger
GateClockDependentMessenger::GateClockDependentMessenger(GateClockDependent* itsClockDependent,
    			     				 const G4String& itsDirectoryName)
: GateNamedObjectMessenger(itsClockDependent,itsDirectoryName)
{ 
  G4String guidance;
  G4String cmdName;


  if (GetClockDependent()->CanBeDisabled()) {
    cmdName = GetDirectoryName()+"enable";
    pEnableCmd = new G4UIcmdWithABool(cmdName,this);
    guidance = G4String("Enables '") + GetDirectoryName() + "'.";
    pEnableCmd->SetGuidance(guidance.c_str());
    pEnableCmd->SetParameterName("flag",true);
    pEnableCmd->SetDefaultValue(true);

    cmdName = GetDirectoryName()+"disable";
    pDisableCmd = new G4UIcmdWithABool(cmdName,this);
    guidance = G4String("Disables '") + GetDirectoryName() + "'.";
    pDisableCmd->SetGuidance(guidance.c_str());
    pDisableCmd->SetParameterName("flag",true);
    pDisableCmd->SetDefaultValue(true);
  }
 ARFcmd = 0;
}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
GateClockDependentMessenger::~GateClockDependentMessenger()
{
  if (GetClockDependent()->CanBeDisabled()) {
    delete pEnableCmd;
    delete pDisableCmd;
  if ( ARFcmd != 0 )
   {
   	 delete ARFcmd;
   	 delete AttachARFSDcmd;
   }


  }
}
//----------------------------------------------------------------------------------------------------------
void GateClockDependentMessenger::SetARFCommands()
{
 G4String cmdName;
 cmdName = GetDirectoryName()+"arf/setARFStage";
 ARFcmd = new G4UIcmdWithAString(cmdName,this);

G4cout << " created command " << cmdName <<G4endl;

 cmdName = GetDirectoryName() +"attachToARFSD";
 AttachARFSDcmd = new G4UIcmdWithoutParameter(cmdName,this);

}

//----------------------------------------------------------------------------------------------------------
// UI command interpreter method
void GateClockDependentMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if ( command==pEnableCmd )
    { GetClockDependent()->Enable(pEnableCmd->GetNewBoolValue(newValue));}   
  else if ( command==pDisableCmd )
    { GetClockDependent()->Enable( !(pDisableCmd->GetNewBoolValue(newValue)) );}     
  else if ( command == ARFcmd )
   { 
   	GateSPECTHeadSystem* theS = dynamic_cast<GateSPECTHeadSystem*>( GetClockDependent() );
    theS->setARFStage( newValue );return;
   }
  else if ( command == AttachARFSDcmd )
    {
     GateVVolume* creator = GateObjectStore::GetInstance()->FindCreator("SPECThead");
     if ( creator != 0 ) creator->AttachARFSD();
     else {
     	   G4cout << " GateObjectCreatorStore : could not find ' "<< GetClockDependent()->GetObjectName()<<" ' "<<G4endl;
     	   G4Exception( "GateClockDependentMessenger::SetNewValue", "SetNewValue", FatalException, "Aborting...");
     	  }
     return;
    }
	else GateNamedObjectMessenger::SetNewValue(command,newValue);
	
}
//----------------------------------------------------------------------------------------------------------
