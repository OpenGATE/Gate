/*----------------------
  Cosmetic cleanup: standardized file comments for cleaner doxygen output


  \brief Class GateOutputMgrMessenger
  \brief By Giovanni.Santin@cern.ch
  \brief $Id: GateOutputMgrMessenger.cc,v 1.2 2002/08/11 15:33:25 dstrul Exp $
*/

#include "GateRTPhantomMgrMessenger.hh"
#include "GateRTPhantomMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRTPhantomMgrMessenger::GateRTPhantomMgrMessenger(GateRTPhantomMgr* PhMgr)
  : GateMessenger(PhMgr->GetName()),
    m_PhMgr(PhMgr)
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

  cmdName = GetDirectoryName()+"insert";

  insertCmd = new G4UIcmdWithAString(cmdName,this);

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRTPhantomMgrMessenger::~GateRTPhantomMgrMessenger()
{
  delete DescribeCmd;
  delete VerboseCmd;
  delete insertCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateRTPhantomMgrMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == VerboseCmd ) {
    m_PhMgr->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  }
  if( command == DescribeCmd ) {
    m_PhMgr->Describe();
  }
  if( command == insertCmd) { m_PhMgr->AddPhantom(newValue); }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



