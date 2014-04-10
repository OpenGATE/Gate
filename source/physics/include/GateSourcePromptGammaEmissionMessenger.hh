/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATESOURCEPROMPTGAMMAEMISSIONMESSENGER_HH
#define GATESOURCEPROMPTGAMMAEMISSIONMESSENGER_HH

#include "G4UImessenger.hh"
#include "G4UIcmdWithAString.hh"
#include "GateConfiguration.h"
#include "GateVSourceMessenger.hh"

//------------------------------------------------------------------------
class GateSourcePromptGammaEmissionMessenger: public GateVSourceMessenger
{
public:
  GateSourcePromptGammaEmissionMessenger(GateSourcePromptGammaEmission* source);
  ~GateSourcePromptGammaEmissionMessenger();
  void SetNewValue(G4UIcommand*, G4String);

private:
  GateSourcePromptGammaEmission * pSourcePromptGammaEmission;
  G4UIcmdWithAString * pSetFilenameCmd;
};
//------------------------------------------------------------------------

#endif // end GATESOURCEPROMPTGAMMAEMISSIONMESSENGER
