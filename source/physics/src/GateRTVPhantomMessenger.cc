/*----------------------
  Cosmetic cleanup: standardized file comments for cleaner doxygen output


  \brief Class GateOutputMgrMessenger
  \brief By Giovanni.Santin@cern.ch
  \brief $Id: GateOutputMgrMessenger.cc,v 1.2 2002/08/11 15:33:25 dstrul Exp $
*/

#include "GateRTVPhantomMessenger.hh"
#include "GateRTVPhantom.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRTVPhantomMessenger::GateRTVPhantomMessenger(GateRTVPhantom* RTVPhantom)
  : GateMessenger(RTVPhantom->GetName()),
    m_RTVPhantom(RTVPhantom)
{ 

  G4String cmdName;


  cmdName = GetDirectoryName() + "setBaseFileName"; 


  SetRTVPhantomCmd = new G4UIcmdWithAString(cmdName,this);
  SetRTVPhantomCmd->SetGuidance("sets the base File Name of all Frames files");
 
  cmdName = GetDirectoryName() + "SetNumberOfFrames"; 


  SetRTVPhantomTPFCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetRTVPhantomTPFCmd->SetGuidance("Sets the number of Frames Files");

  cmdName = GetDirectoryName() + "setHeaderFileName";

  SetRTVPhantomHeaderFileCmd = new G4UIcmdWithAString(cmdName,this);
  SetRTVPhantomHeaderFileCmd->SetGuidance("sets the Header File Name of all Frames files");


  cmdName = GetDirectoryName() + "SetTimePerFrame";


  SetTPFCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  SetTPFCmd->SetGuidance("Sets the number of Frames Files");

  cmdName = GetDirectoryName() + "SetAttenuationMapAsActivityMap";


  SetAttAsActCmd = new G4UIcmdWithoutParameter(cmdName,this);
  SetAttAsActCmd->SetGuidance("Sets the Attenuation Map to be the same as the Activity Map for each frame");

  cmdName = GetDirectoryName() + "SetActivityMapAsAttenuationMap";


  SetActAsAttCmd = new G4UIcmdWithoutParameter(cmdName,this);
  SetActAsAttCmd->SetGuidance("Sets the Activity Map to be the same as the Attenuation Map for each frame");
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRTVPhantomMessenger::~GateRTVPhantomMessenger()
{
    delete SetRTVPhantomTPFCmd;
    delete SetRTVPhantomCmd;
    delete SetRTVPhantomHeaderFileCmd;
    delete SetTPFCmd;
    delete SetActAsAttCmd;
    delete SetAttAsActCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateRTVPhantomMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 


 if ( command==SetTPFCmd )
{
G4double TPF = SetTPFCmd->GetNewDoubleValue(newValue);
m_RTVPhantom->SetTPF( TPF ) ;
      G4cout << " Real Time RTVPhantom Time Per Frames set to " << m_RTVPhantom->GetTPF() << G4endl;
return;  }


if( command==SetRTVPhantomHeaderFileCmd )
{      m_RTVPhantom->SetHeaderFileName(newValue);
return;  }

 
 if ( command==SetRTVPhantomTPFCmd )
{
G4int NbOfFrames = (G4int)(SetRTVPhantomTPFCmd->GetNewIntValue(newValue));
m_RTVPhantom->SetNbOfFrames( NbOfFrames ) ;
      G4cout << " Real Time RTVPhantom Number of Frames set to " << m_RTVPhantom->GetNbOfFrames() << G4endl;
return;  }

if( command==SetRTVPhantomCmd )
{      m_RTVPhantom->SetBaseFileName(newValue);
return;  }   

if( command == SetActAsAttCmd )
{      m_RTVPhantom->SetActAsAtt();
return;  }   
if( command == SetAttAsActCmd )
{      m_RTVPhantom->SetAttAsAct();
return;  }  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
