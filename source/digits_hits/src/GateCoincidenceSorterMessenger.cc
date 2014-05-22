/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceSorterMessenger.hh"

#include "GateCoincidenceSorter.hh"
//#include "GateSystemListManager.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"

GateCoincidenceSorterMessenger::GateCoincidenceSorterMessenger(GateCoincidenceSorter* itsCoincidenceSorter)
    : GateClockDependentMessenger(itsCoincidenceSorter)
{

  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setWindow";
  windowCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  windowCmd->SetGuidance("Set time-window for coincidence");
  windowCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "setOffset";
  offsetCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  offsetCmd->SetGuidance("Set time offset for delay coincidences");
  offsetCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "setOffsetJitter";
  offsetJitterCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  offsetJitterCmd->SetGuidance("Set time offset for delay coincidences");
  offsetJitterCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "setWindowJitter";
  windowJitterCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  windowJitterCmd->SetGuidance("Set time-window for coincidence");
  windowJitterCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName()+"minSectorDifference";
  minSectorDiffCmd = new G4UIcmdWithAnInteger(cmdName.c_str(),this);
  minSectorDiffCmd->SetGuidance("Set the minimum sector difference for valid coincidences.");
  minSectorDiffCmd->SetParameterName("diff",false);
  minSectorDiffCmd->SetRange("diff>=1");

  cmdName = GetDirectoryName()+"setDepth";
  setDepthCmd = new G4UIcmdWithAnInteger(cmdName.c_str(),this);
  setDepthCmd->SetGuidance("Set the depth of system-level for coincidences.");
  setDepthCmd->SetParameterName("depth",false);
  setDepthCmd->SetRange("depth>=1");

  cmdName = GetDirectoryName()+"setInputName";
  SetInputNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetInputNameCmd->SetGuidance("Set the name of the input pulse channel");
  SetInputNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"MultiplesPolicy";
  MultiplePolicyCmd = new G4UIcmdWithAString(cmdName,this);
  MultiplePolicyCmd->SetGuidance("How to treat multiples coincidences");
  MultiplePolicyCmd->SetCandidates("killAll takeAllGoods killAllIfMultipleGoods takeWinnerOfGoods takeWinnerIfIsGood takeWinnerIfAllAreGoods keepAll keepIfAnyIsGood keepIfOnlyOneGood keepIfAllAreGoods");

  cmdName = GetDirectoryName()+"allPulseOpenCoincGate";
  AllPulseOpenCoincGateCmd = new G4UIcmdWithABool(cmdName,this);
  AllPulseOpenCoincGateCmd->SetGuidance("Specify if a given pulse can be part of two coincs");

}


GateCoincidenceSorterMessenger::~GateCoincidenceSorterMessenger()
{
  delete windowCmd;
  delete offsetCmd;
  delete minSectorDiffCmd;
  delete SetInputNameCmd;
  delete MultiplePolicyCmd;
  delete AllPulseOpenCoincGateCmd;
}


void GateCoincidenceSorterMessenger::SetNewValue(G4UIcommand* aCommand, G4String newValue)
{
  if ( aCommand==windowCmd )
    { GetCoincidenceSorter()->SetWindow(windowCmd->GetNewDoubleValue(newValue)); }
  else if( aCommand == windowJitterCmd )
    { GetCoincidenceSorter()->SetWindowJitter(windowJitterCmd->GetNewDoubleValue(newValue)); }
  else if( aCommand == offsetCmd )
    { GetCoincidenceSorter()->SetOffset(offsetCmd->GetNewDoubleValue(newValue)); }
  else if( aCommand == offsetJitterCmd )
    { GetCoincidenceSorter()->SetOffsetJitter(offsetJitterCmd->GetNewDoubleValue(newValue)); }
  else if( aCommand == minSectorDiffCmd )
    { GetCoincidenceSorter()->SetMinSectorDifference(minSectorDiffCmd->GetNewIntValue(newValue)); }
  else if( aCommand == setDepthCmd )
    { GetCoincidenceSorter()->SetDepth(setDepthCmd->GetNewIntValue(newValue)); }
  else if (aCommand == SetInputNameCmd)
    {
     GetCoincidenceSorter()->SetInputName(newValue);
     GetCoincidenceSorter()->SetSystem(newValue); //! Attach to the suitable system from the digitizer m_systemList (multi-system approach)
    }
  else if (aCommand == MultiplePolicyCmd)
    { GetCoincidenceSorter()->SetMultiplesPolicy(newValue); }
  else if (aCommand == AllPulseOpenCoincGateCmd)
    { GetCoincidenceSorter()->SetAllPulseOpenCoincGate(AllPulseOpenCoincGateCmd->GetNewBoolValue(newValue)); }
  else
    GateClockDependentMessenger::SetNewValue(aCommand,newValue);
}
