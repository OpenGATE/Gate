/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateExtendedAnalysisMessenger.hh"
#include "GateExtendedAnalysis.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateObjectStore.hh"

GateExtendedAnalysisMessenger::GateExtendedAnalysisMessenger(GateExtendedAnalysis* gateExtendedAnalysis)
  : GateOutputModuleMessenger(gateExtendedAnalysis)
  , m_GateExtendedAnalysis(gateExtendedAnalysis)
{
}

void GateExtendedAnalysisMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
 GateOutputModuleMessenger::SetNewValue(command,newValue);
}
