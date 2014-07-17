/*
 * GateParaboloidMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEPARABOLOIDMESSENGER_HH_
#define GATEPARABOLOIDMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateParaboloid;

class GateParaboloidMessenger: public GateVolumeMessenger
{
public:
  GateParaboloidMessenger(GateParaboloid* itsCreator);
  ~GateParaboloidMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateParaboloid* GetParaboloidCreator()
    { return (GateParaboloid*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* ParaboloidPositiveRCmd;
  G4UIcmdWithADoubleAndUnit* ParaboloidNegativeRCmd;
  G4UIcmdWithADoubleAndUnit* ParaboloidZLengthCmd;
};

#endif /* GATEPARABOLOIDMESSENGER_HH_ */
