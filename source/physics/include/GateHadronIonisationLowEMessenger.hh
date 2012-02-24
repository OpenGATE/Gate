/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEHADIONLOWEMESSENGER_HH
#define GATEHADIONLOWEMESSENGER_HH


#include "GateEMStandardProcessMessenger.hh"
#include "GateVProcess.hh"

#include "G4hLowEnergyIonisation.hh"
#include "G4UIcmdWithAString.hh"


class GateHadronIonisationLowEMessenger:public GateEMStandardProcessMessenger
{
public:
  GateHadronIonisationLowEMessenger(GateVProcess *pb);
  virtual ~GateHadronIonisationLowEMessenger();

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  G4UIcmdWithAString * pSetNuclearStopping;
  G4UIcmdWithAString * pUnsetNuclearStopping;
};

#endif /* end #define GATEHADIONLOWEMESSENGER_HH */
