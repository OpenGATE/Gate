/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTORMESSENGER_HH
#define GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTORMESSENGER_HH

#include "GateConfiguration.h"
#include "GateActorMessenger.hh"

#include <G4UIcmdWithAnInteger.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>

class GatePromptGammaSpectrumDistributionActor;

//-----------------------------------------------------------------------------
class GatePromptGammaSpectrumDistributionActorMessenger: public GateActorMessenger
{
public:

  GatePromptGammaSpectrumDistributionActorMessenger(GatePromptGammaSpectrumDistributionActor*);
  ~GatePromptGammaSpectrumDistributionActorMessenger();

  void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);
  GatePromptGammaSpectrumDistributionActor* pActor;

  G4UIcmdWithADoubleAndUnit * pProtonEMinCmd;
  G4UIcmdWithADoubleAndUnit * pProtonEMaxCmd;
  G4UIcmdWithADoubleAndUnit * pGammaEMinCmd;
  G4UIcmdWithADoubleAndUnit * pGammaEMaxCmd;
  G4UIcmdWithAnInteger * pProtonNbBinsCmd;
  G4UIcmdWithAnInteger * pGammaNbBinsCmd;

};
//-----------------------------------------------------------------------------

#endif
