/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCoincidencePulseProcessorChainMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"

#include "GateVCoincidencePulseProcessor.hh"
#include "GateCoincidencePulseProcessorChain.hh"

#include "GateCoincidenceDeadTime.hh"
#include "GateCoincidenceTimeDiffSelector.hh"
#include "GateCoincidenceGeometrySelector.hh"
#include "GateCoincidenceBuffer.hh"
#include "GateCoincidenceMultiplesKiller.hh"
#include "GateTriCoincidenceSorter.hh" //mhadi_add
#include "GateCCCoincidenceSequenceRecon.hh"//AE

GateCoincidencePulseProcessorChainMessenger::GateCoincidencePulseProcessorChainMessenger(GateCoincidencePulseProcessorChain* itsProcessorChain)
:GateListMessenger(itsProcessorChain)
{ 
  pInsertCmd->SetCandidates(DumpMap());

  G4String cmdName;

  cmdName = GetDirectoryName()+"addInputName";
  AddInputNameCmd = new G4UIcmdWithAString(cmdName,this);
  AddInputNameCmd->SetGuidance("Add a name for the input pulse channel");
  AddInputNameCmd->SetParameterName("Name",false);
  cmdName = GetDirectoryName()+"usePriority";
  usePriorityCmd = new G4UIcmdWithABool(cmdName,this);
  usePriorityCmd->SetGuidance("Does it use insertion order in case of different coinc arrived at the same time ?");
  usePriorityCmd->SetParameterName("Use",false);
}




GateCoincidencePulseProcessorChainMessenger::~GateCoincidencePulseProcessorChainMessenger()
{
  delete AddInputNameCmd;
  delete usePriorityCmd;
}




void GateCoincidencePulseProcessorChainMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if (command == AddInputNameCmd) 
  { GetProcessorChain()->GetInputNames().push_back(newValue);  //mhadi_modif
    GetProcessorChain()->SetSystem(newValue); } //mhadi
  else if (command == usePriorityCmd) 
    { GetProcessorChain()->SetNoPriority(!usePriorityCmd->GetNewBoolValue(newValue)); }
  else
    GateListMessenger::SetNewValue(command,newValue);
}




const G4String& GateCoincidencePulseProcessorChainMessenger::DumpMap()
{
   static G4String theList = "deadtime  sequenceRecon timeDiffSelector geometrySelector buffer multiplesKiller triCoincProcessor";//mhadi_modif
  return theList;
}



void GateCoincidencePulseProcessorChainMessenger::DoInsertion(const G4String& childTypeName)
{
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(childTypeName);
    
  AvoidNameConflicts();

  GateVCoincidencePulseProcessor* newProcessor=0;

  G4String newInsertionName = GetProcessorChain()->MakeElementName(GetNewInsertionBaseName());

 /* if (childTypeName=="deadtime")
    newProcessor = new GateCoincidenceDeadTime(GetProcessorChain(),newInsertionName);
  else
  */
  if (childTypeName=="sequenceRecon")
    newProcessor = new GateCCCoincidenceSequenceRecon(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="timeDiffSelector")
    newProcessor = new GateCoincidenceTimeDiffSelector(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="geometrySelector")
    newProcessor = new GateCoincidenceGeometrySelector(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="buffer")
    newProcessor = new GateCoincidenceBuffer(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="multiplesKiller")
    newProcessor = new GateCoincidenceMultiplesKiller(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="triCoincProcessor") //mhadi_add
     newProcessor = new GateTriCoincidenceSorter(GetProcessorChain(),newInsertionName);//mhadi_add
  else {
    G4cout << "Pulse-processor type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    return;
  }
  
  GetProcessorChain()->InsertProcessor(newProcessor);
  SetNewInsertionBaseName("");
}


G4bool GateCoincidencePulseProcessorChainMessenger::CheckNameConflict(const G4String& name)
{
  // Check whether an object with the same name already exists in the list
  return ( GetListManager()->FindElement( GetListManager()->GetObjectName() + "/" + name ) != 0 ) ;
}





