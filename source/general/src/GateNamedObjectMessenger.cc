/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateNamedObjectMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateNamedObject.hh"


//-------------------------------------------------------------------------------------------
// Constructor
// The flag 'itsFlagDescribe' tells whether this messenger should propose a command "Describe"
// The flag 'flagCreateDirectory' is passed to the base-class GateMessenger
GateNamedObjectMessenger::GateNamedObjectMessenger(GateNamedObject* itsNamedVolume,
    			                           const G4String& itsDirectoryName)
: GateMessenger(itsDirectoryName.empty() ? itsNamedVolume->GetObjectName() : itsDirectoryName ),
  pNamedVolume(itsNamedVolume)
{ 
  G4String guidance;
  G4String cmdName;
 
  cmdName = GetDirectoryName()+"describe";
  pDescribeCmd = new G4UIcmdWithoutParameter(cmdName,this);
  guidance = G4String("Print-out a description of the object.");
  pDescribeCmd->SetGuidance(guidance.c_str());

  cmdName = GetDirectoryName() + "verbose";
  pVerbosityCmd = new G4UIcmdWithAnInteger(cmdName,this);
  pVerbosityCmd->SetGuidance("Set verbosity level for system");
  pVerbosityCmd->SetGuidance(" 0 : Silent");
  pVerbosityCmd->SetGuidance(" 1 : Limited information");
  pVerbosityCmd->SetGuidance(" 2 : Detailed information");
  pVerbosityCmd->SetParameterName("level",false);
  pVerbosityCmd->SetRange("level>=0 && level <=6");

}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
GateNamedObjectMessenger::~GateNamedObjectMessenger()
{
  delete pDescribeCmd;
  delete pVerbosityCmd;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
// UI command interpreter method
void GateNamedObjectMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 

  if ( command == pDescribeCmd )
  { 
     pNamedVolume->Describe(); 
  }   
  else if ( command == pVerbosityCmd)
  {    
      pNamedVolume->SetVerbosity(pVerbosityCmd->GetNewIntValue(newValue));
  }
  else
    GateMessenger::SetNewValue(command,newValue);
  
}
//-------------------------------------------------------------------------------------------
