/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFDataToRootMessenger.hh"
#include "GateARFDataToRoot.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateARFDataToRootMessenger::GateARFDataToRootMessenger(GateARFDataToRoot* GateARFDataToRoot)
  : GateOutputModuleMessenger(GateARFDataToRoot)
  , m_GateARFDataToRoot(GateARFDataToRoot)
{ 
  G4String cmdName;

  //G4cout << " created directory " << GetDirectoryName()<<G4endl;

  cmdName = GetDirectoryName()+"setFileName";

  setARFDataFilecmd = new G4UIcmdWithAString(cmdName,this);
  setARFDataFilecmd->SetGuidance("sets the ARF Data Root File Name");
  cmdName = GetDirectoryName()+"setProjectionPlane";
  setDepth= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  setDepth->SetGuidance("sets the YZ projection plane relative to the ARF device center");

  cmdName = GetDirectoryName()+"applyToDRFData";
  smoothDRFcmd = new G4UIcmdWithAString(cmdName,this);


}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateARFDataToRootMessenger::~GateARFDataToRootMessenger()
{

delete setARFDataFilecmd;
delete setDepth ;
delete smoothDRFcmd ;
    
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateARFDataToRootMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 

 if ( command == smoothDRFcmd ) 
 {   if ( newValue == "smoothness" )
     {m_GateARFDataToRoot->setDRFDataprojectionmode(0);return;}
     if ( newValue == "line-projection" ){m_GateARFDataToRoot->setDRFDataprojectionmode(1);return;}
     if ( newValue == "orthogonal-projection" ){m_GateARFDataToRoot->setDRFDataprojectionmode(2);return;}
  G4cout << " GateARFSimuSDMessenger::SetNewValue ::: UNKNOWN parameter "<<newValue<<". Ignored DRF Data projection Mode. Set To Default : smoothness "<<G4endl;
    return;
 }

if ( command == setDepth )
  {
   m_GateARFDataToRoot->SetProjectionPlane(  setDepth->GetNewDoubleValue( newValue  ) );
  return;
  }



  if ( command == setARFDataFilecmd )
  {
   m_GateARFDataToRoot->SetARFDataRootFileName( newValue );return;
  }

    GateOutputModuleMessenger::SetNewValue(command,newValue); 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
