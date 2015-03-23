/*
 * GateTwistedTrdMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDTRDMESSENGER_HH_
#define GATETWISTEDTRDMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTwistedTrd;

class GateTwistedTrdMessenger: public GateVolumeMessenger
{
public:
  GateTwistedTrdMessenger(GateTwistedTrd* itsCreator);
  ~GateTwistedTrdMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateTwistedTrd* GetTwistedTrdCreator()
    { return (GateTwistedTrd*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* TwistedTrdX1LengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrdX2LengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrdY1LengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrdY2LengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrdZLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrdTwistAngleCmd;
};



#endif /* GATETWISTEDTRDMESSENGER_HH_ */
