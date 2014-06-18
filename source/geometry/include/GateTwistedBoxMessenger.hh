/*
 * GateTwistedBoxMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDBOXMESSENGER_HH_
#define GATETWISTEDBOXMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTwistedBox;

class GateTwistedBoxMessenger: public GateVolumeMessenger
{
public:
  GateTwistedBoxMessenger(GateTwistedBox* itsCreator);
  ~GateTwistedBoxMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateTwistedBox* GetTwistedBoxCreator()
    { return (GateTwistedBox*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* TwistedBoxXLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedBoxYLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedBoxZLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedBoxTwistAngleCmd;
};



#endif /* GATETWISTEDBOXMESSENGER_HH_ */
