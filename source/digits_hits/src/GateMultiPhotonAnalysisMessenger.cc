/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateMultiPhotonAnalysisMessenger.hh"

#include "GateMultiPhotonAnalysis.hh"

#include "G4UIcmdWithAString.hh"

GateMultiPhotonAnalysisMessenger::GateMultiPhotonAnalysisMessenger(
    GateMultiPhotonAnalysis *gateMultiPhotonAnalysis)
    : GateOutputModuleMessenger(gateMultiPhotonAnalysis),
      m_gateMultiPhotonAnalysis(gateMultiPhotonAnalysis),
      m_setMissingTrajectoryPolicyCmd(0) {
  G4String cmdName = GetDirectoryName() + "setMissingTrajectoryPolicy";
  m_setMissingTrajectoryPolicyCmd = new G4UIcmdWithAString(cmdName, this);
  m_setMissingTrajectoryPolicyCmd->SetGuidance("Set policy for missing trajectory container handling.");
  m_setMissingTrajectoryPolicyCmd->SetGuidance("Options: strict resilient");
  m_setMissingTrajectoryPolicyCmd->SetCandidates("strict resilient");
  m_setMissingTrajectoryPolicyCmd->SetParameterName("policy", false);
}

GateMultiPhotonAnalysisMessenger::~GateMultiPhotonAnalysisMessenger() {
  delete m_setMissingTrajectoryPolicyCmd;
}

void GateMultiPhotonAnalysisMessenger::SetNewValue(G4UIcommand *command, G4String newValue) {
  if (command == m_setMissingTrajectoryPolicyCmd) {
    m_gateMultiPhotonAnalysis->SetMissingTrajectoryPolicyFromString(newValue);
    return;
  }

  GateOutputModuleMessenger::SetNewValue(command, newValue);
}
