/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceMgrMessenger.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//-------------------------------------------------------------------------------
GateSourceMgrMessenger::GateSourceMgrMessenger(GateSourceMgr* sourceMgr)
{ 
  m_sourceMgr = sourceMgr;

  GateSourceDir = new G4UIdirectory("/gate/source/");
  GateSourceDir->SetGuidance("GATE source manager control.");

  SelectSourceCmd = new G4UIcmdWithAString("/gate/source/selectSource",this);
  SelectSourceCmd->SetGuidance("Select a source by name");
  SelectSourceCmd->SetGuidance("1. Source name");

  AddSourceCmd = new GateUIcmdWithAVector<G4String>("/gate/source/addSource",this);
  AddSourceCmd->SetGuidance("Add a source");
  AddSourceCmd->SetGuidance("1. Source name");
  AddSourceCmd->AvailableForStates(G4State_Idle,G4State_GeomClosed,G4State_EventProc);

  RemoveSourceCmd = new G4UIcmdWithAString("/gate/source/removeSource",this);
  RemoveSourceCmd->SetGuidance("Remove a source");
  RemoveSourceCmd->SetGuidance("1. Source name");

  ListSourcesCmd = new G4UIcmdWithoutParameter("/gate/source/list",this);
  ListSourcesCmd->SetGuidance("List of the sources");

  VerboseCmd = new G4UIcmdWithAnInteger("/gate/source/verbose",this);
  VerboseCmd->SetGuidance("Set GATE event action verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  // UseAutoWeightCmd = new G4UIcmdWithAnInteger("/gate/source/useSameNumberOfParticlesPerRun",this);
  //UseAutoWeightCmd->SetGuidance("The number of particles per source is the same. The weight is set automatically.");
  // UseAutoWeightCmd->SetParameterName("Number of particles",false);

}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
GateSourceMgrMessenger::~GateSourceMgrMessenger()
{
  delete AddSourceCmd;
  delete RemoveSourceCmd;
  delete SelectSourceCmd;
  delete ListSourcesCmd;
  delete VerboseCmd;
  delete GateSourceDir;
  //delete UseAutoWeightCmd;
}
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
void GateSourceMgrMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == VerboseCmd ) {
    m_sourceMgr->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  } else if( command == SelectSourceCmd ) {
    m_sourceMgr->SelectSourceByName(newValue);
  } else if( command == AddSourceCmd ) {
    m_sourceMgr->AddSource(AddSourceCmd->GetNewVectorValue(newValue));
  } else if( command == RemoveSourceCmd ) {
    m_sourceMgr->RemoveSource(newValue);
  } else if( command == ListSourcesCmd ) {
    m_sourceMgr->ListSources();
  }/* else if( command == UseAutoWeightCmd ) {
    m_sourceMgr->SetNTot(UseAutoWeightCmd->GetNewIntValue(newValue));
    }*/
}
//-------------------------------------------------------------------------------


