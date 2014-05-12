/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGENERALHADPROCESSMESSENGER_HH
#define GATEGENERALHADPROCESSMESSENGER_HH


#include "GateVProcessMessenger.hh"
#include "GateVProcess.hh"


class GateHadronicStandardProcessMessenger:public GateVProcessMessenger
{
public:
  GateHadronicStandardProcessMessenger(GateVProcess *pb);
  virtual ~GateHadronicStandardProcessMessenger(){};

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);
};

#endif /* end #define GATEGENRALHADPROCESSMESSENGER_HH */
