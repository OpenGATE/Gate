/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGENERALEMPROCESSMESSENGER_HH
#define GATEGENERALEMPROCESSMESSENGER_HH


#include "GateVProcessMessenger.hh"
#include "GateVProcess.hh"

#include "G4UIcmdWithAString.hh"
#include "GateUIcmdWithAStringADoubleAndADoubleWithUnit.hh"
#include "GateUIcmdWithAStringAndADouble.hh"

class GateEMStandardProcessMessenger:public GateVProcessMessenger
{
public:
  GateEMStandardProcessMessenger(GateVProcess *pb);
  virtual ~GateEMStandardProcessMessenger();

  virtual void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  GateUIcmdWithAStringADoubleAndADoubleWithUnit * pSetStepFctCmd;
  GateUIcmdWithAStringAndADouble * pSetLinearlossLimit;

};

#endif /* end #define GATEGENRALEMPROCESSMESSENGER_HH */
