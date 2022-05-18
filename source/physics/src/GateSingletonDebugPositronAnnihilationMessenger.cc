/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



#include "GateSingletonDebugPositronAnnihilationMessenger.hh"
#include "GateSingletonDebugPositronAnnihilation.hh"

//-----------------------------------------------------------------------------
GateSingletonDebugPositronAnnihilationMessenger::GateSingletonDebugPositronAnnihilationMessenger()
{
  G4String guidance;
  pActiveDebugFlagCmd = new G4UIcmdWithABool("/gate/source/setDebugPositronAnnihilationFlag",this);
  guidance = "Activation of the debug flag for positron annihilation";
  pActiveDebugFlagCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSingletonDebugPositronAnnihilationMessenger::~GateSingletonDebugPositronAnnihilationMessenger()
{
  delete pActiveDebugFlagCmd;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateSingletonDebugPositronAnnihilationMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pActiveDebugFlagCmd){
    auto sDebugPositronAnnihilation = GateSingletonDebugPositronAnnihilation::GetInstance();
    sDebugPositronAnnihilation->SetDebugFlag(pActiveDebugFlagCmd->GetNewBoolValue(param));
  }
}
//-----------------------------------------------------------------------------

