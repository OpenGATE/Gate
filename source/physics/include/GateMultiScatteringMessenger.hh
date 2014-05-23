/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGENERALSCATTERINGPROCESSMESSENGER_HH
#define GATEGENERALSCATTERINGPROCESSMESSENGER_HH


#include "GateEMStandardProcessMessenger.hh"
#include "GateVProcess.hh"

#include "G4UIcmdWithAString.hh"
#include "GateUIcmdWith2String.hh"
#include "GateUIcmdWithAStringAndADouble.hh"

class GateMultiScatteringMessenger:public GateEMStandardProcessMessenger
{
public:
  GateMultiScatteringMessenger(GateVProcess *pb);
  virtual ~GateMultiScatteringMessenger();

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  GateUIcmdWith2String * pSetDistanceToBoundary;

};

#endif 
