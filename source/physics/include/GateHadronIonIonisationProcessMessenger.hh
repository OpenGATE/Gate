/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEHADRONIONIONIPROCESSMESSENGER_HH
#define GATEHADRONIONIONIPROCESSMESSENGER_HH

#include "globals.hh"

#include "GateEMStandardProcessMessenger.hh"
#include "GateVProcess.hh"

#include "G4UIcmdWithAString.hh"

class GateHadronIonIonisationProcessMessenger:public GateEMStandardProcessMessenger
{
public:
  GateHadronIonIonisationProcessMessenger(GateVProcess *pb);
  ~GateHadronIonIonisationProcessMessenger();

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  G4UIcmdWithAString * pSetNuclearStopping;
  G4UIcmdWithAString * pUnsetNuclearStopping;
};

#endif /* end #define GATEHADRONIONIONIPROCESSMESSENGER_HH */
