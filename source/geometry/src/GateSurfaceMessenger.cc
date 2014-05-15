/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateSurfaceMessenger.hh"


GateSurfaceMessenger::GateSurfaceMessenger(GateSurface* itsSurface) :
  GateClockDependentMessenger(itsSurface, itsSurface->GetInserter1()->GetObjectName()+"/surfaces/"+itsSurface->GetObjectName()),
  m_setSurfaceCmd(0)
{
  G4String cmdName = GetDirectoryName()+"setSurface";
  m_setSurfaceCmd = new G4UIcmdWithAString(cmdName,this);
  G4String guidance = G4String("Set the optical properties of the surface '") + GetDirectoryName() + "'.";
  m_setSurfaceCmd->SetGuidance(guidance.c_str());
}

GateSurfaceMessenger::~GateSurfaceMessenger()
{
  if (m_setSurfaceCmd) delete m_setSurfaceCmd;
}

void GateSurfaceMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command==m_setSurfaceCmd)
  { GetSurface()->SetOpticalSurfaceName(newValue);}
  else GateClockDependentMessenger::SetNewValue(command, newValue);
}

#endif
