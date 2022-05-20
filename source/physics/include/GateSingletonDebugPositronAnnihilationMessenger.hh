/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GATESINGLETONDEBUGPOSITRONANNIHILATIONMESSENGER_HH
#define GATESINGLETONDEBUGPOSITRONANNIHILATIONMESSENGER_HH


#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UImessenger.hh"

class GateSingletonDebugPositronAnnihilationMessenger:public G4UImessenger
{
public:
  GateSingletonDebugPositronAnnihilationMessenger();
  virtual ~GateSingletonDebugPositronAnnihilationMessenger();

  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  G4UIcmdWithABool * pActiveDebugFlagCmd;
  G4UIcmdWithAString * pOutputFileCmd;

};

#endif 
