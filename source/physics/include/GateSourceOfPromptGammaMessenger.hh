/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATESOURCEPROMPTGAMMAEMISSIONMESSENGER_HH
#define GATESOURCEPROMPTGAMMAEMISSIONMESSENGER_HH

#include "GateConfiguration.h"
#include "GateImageActorMessenger.hh"
#include "GateVSourceMessenger.hh"
#include "GateSourceOfPromptGamma.hh"

class GateSourceOfPromptGamma;

//------------------------------------------------------------------------
class GateSourceOfPromptGammaMessenger: public GateVSourceMessenger
{
public:
  GateSourceOfPromptGammaMessenger(GateSourceOfPromptGamma* source);
  ~GateSourceOfPromptGammaMessenger();
  void SetNewValue(G4UIcommand*, G4String);

private:
  GateSourceOfPromptGamma * pSourceOfPromptGamma;
  G4UIcmdWithAString * pSetFilenameCmd;
  G4UIcmdWithABool * pSetTofCmd;
};
//------------------------------------------------------------------------

#endif // end GATESOURCEPROMPTGAMMAEMISSIONMESSENGER
