/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSystemComponentMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateSystemComponent.hh"
#include "GateObjectStore.hh"
#include "GateVSystem.hh"
#include "GateVVolume.hh"


// Constructor
GateSystemComponentMessenger::GateSystemComponentMessenger(GateSystemComponent* itsSystemComponent)
   : GateClockDependentMessenger( itsSystemComponent, FabricateDirName(itsSystemComponent))
{ 
  
  G4String guidance;
  G4String cmdName;

  SetDirectoryGuidance(G4String("Controls the system-component '") + itsSystemComponent->GetObjectName() + "'" );

  cmdName = GetDirectoryName() + "attach";
    
  AttachCmd = new G4UIcmdWithAString(cmdName,this);
  AttachCmd->SetGuidance("Attach a new volume to the system-component.");
  AttachCmd->SetParameterName("choice",false);

  //  create a min sector difference for each defined rsector only


GateSystemComponent* motherComponent = itsSystemComponent->GetMotherComponent();
G4String motherComponentName = G4String("NotDefined");
minSectorDiffCmd = 0;
setInCoincidenceWithCmd = 0; setRingIDCmd = 0;
if ( motherComponent != 0 )
{
  motherComponentName =motherComponent->GetObjectName();
  size_t pos = motherComponentName.rfind("/");
  motherComponentName = motherComponentName.substr( pos+1 );
}

if ( motherComponentName == "base" )
{
  itsSystemComponent->SetminSectorDiff( 1 ); // for rsector default is one Sector Difference
  cmdName = GetDirectoryName()+"setMinSectorDifference";

  minSectorDiffCmd = new G4UIcmdWithAnInteger(cmdName.c_str(),this);
  minSectorDiffCmd->SetGuidance("Set the minimum sector difference for valid coincidences inside a given rsector.");
  minSectorDiffCmd->SetParameterName("diff",false);
  minSectorDiffCmd->SetRange("diff>=1");
  cmdName = GetDirectoryName()+"setInCoincidenceWith";

  setInCoincidenceWithCmd = new G4UIcmdWithAString(cmdName.c_str(),this);
  setInCoincidenceWithCmd->SetGuidance("enables coincidences between two different Rsectors .");

  cmdName = GetDirectoryName()+"setRingID";

  setRingIDCmd = new G4UIcmdWithAnInteger(cmdName.c_str(),this);

}
}



// Destructor
GateSystemComponentMessenger::~GateSystemComponentMessenger()
{
  delete AttachCmd;
	if ( minSectorDiffCmd != 0 ) delete minSectorDiffCmd;
    if ( setInCoincidenceWithCmd  != 0 ) delete setInCoincidenceWithCmd;
    if ( setRingIDCmd != 0 ) delete setRingIDCmd;
}



// UI command interpreter method
void GateSystemComponentMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
	if ( command == minSectorDiffCmd ) {GetSystemComponent()->SetminSectorDiff( minSectorDiffCmd->GetNewIntValue(newValue) );
                                                           return;}

    if ( command == setInCoincidenceWithCmd ) {GetSystemComponent()->setInCoincidenceWith( newValue );
                                                                       return;}

   if ( command == setRingIDCmd ) { GetSystemComponent()->SetRingID( setRingIDCmd->GetNewIntValue(newValue) ); return;}
 
  if( command==AttachCmd )
    { 
    AddCreator(newValue); }   
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}





// Method to apply the UI command 'attach'
// Finds an creator from its name and attaches this creator to the system component
void GateSystemComponentMessenger::AddCreator(const G4String& creatorName)
{ 
  
  // Find the creator from the creator's name
  GateVVolume* anCreator = GateObjectStore::GetInstance()->FindCreator(creatorName);
  
//  G4cout << " GateObjectStore::GetInstance()->FindCreator(creatorName) "  << G4endl;
  
  // If an creator was found, ask the system component to perform the attachement  
  if (anCreator) 
    {  GetSystemComponent()->SetCreator(anCreator); }
  else
    G4cerr  << "[GateSystemComponentMessenger]: " << G4endl
      	    << "could not find a volume creator for the name '" << creatorName << "'" << G4endl
	    << "Attachment request will be ignored!" << G4endl << G4endl;
}

// Next method was added for the multi-system approach 
G4String GateSystemComponentMessenger::FabricateDirName(const GateSystemComponent* component)
{
   G4String dirName = "";

   G4String sysComponentName = component->GetObjectName();
   size_t pos = sysComponentName.find("/", 8);
   sysComponentName = sysComponentName.substr(pos);
   if(sysComponentName.compare("/base") == 0)
      return dirName;
   
   G4String systemOwnName = component->GetSystem()->GetOwnName();

   if(!systemOwnName.empty())
      dirName = "systems/" + systemOwnName;
   else
      dirName += component->GetSystem()->GetName();

   dirName += sysComponentName;
   
   return dirName;
}

