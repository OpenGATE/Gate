/*----------------------
  Cosmetic cleanup: standardized file comments for cleaner doxygen output


  \brief Class GateOutputMgrMessenger
  \brief By Giovanni.Santin@cern.ch
  \brief $Id: GateOutputMgrMessenger.cc,v 1.2 2002/08/11 15:33:25 dstrul Exp $
*/

#include "GateRTPhantomMessenger.hh"
#include "GateRTPhantom.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRTPhantomMessenger::GateRTPhantomMessenger(GateRTPhantom* Ph)
  : GateMessenger(Ph->GetName()),
    m_Ph(Ph)
{ 

  G4String cmdName;


  cmdName = GetDirectoryName()+"describe";



  DescribeCmd = new G4UIcmdWithoutParameter(cmdName,this);
  DescribeCmd->SetGuidance("List of the output manager properties");

  cmdName = GetDirectoryName()+"verbose";

  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set GATE output manager verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  cmdName = GetDirectoryName()+"AttachTo";

  attachCmd = new G4UIcmdWithAString(cmdName,this);

  cmdName = GetDirectoryName()+"AttachToSource";

  attachSCmd = new G4UIcmdWithAString(cmdName,this);

  cmdName = GetDirectoryName()+"disable";

  DisableCmd = new G4UIcmdWithoutParameter(cmdName,this);
  DisableCmd->SetGuidance("List of the output manager properties");


}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRTPhantomMessenger::~GateRTPhantomMessenger()
{
  delete DescribeCmd;
  delete VerboseCmd;
  delete attachCmd;
  delete DisableCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateRTPhantomMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 

  if( command == VerboseCmd ) {
    m_Ph->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
    return;}
  if( command == DescribeCmd ) {
    m_Ph->Describe();
    return;
  }
  if( command == attachCmd) { m_Ph->AttachToGeometry(newValue);return; }

  if( command == attachSCmd) { m_Ph->AttachToSource(newValue);return; }

  if( command == DisableCmd ) {
    m_Ph->Disable();
    return;
  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



