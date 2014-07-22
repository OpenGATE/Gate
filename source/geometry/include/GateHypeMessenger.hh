/*
 * GateHypeMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEHYPEMESSENGER_HH_
#define GATEHYPEMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateHype;

class GateHypeMessenger: public GateVolumeMessenger
{
public:
  GateHypeMessenger(GateHype* itsCreator);
  ~GateHypeMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateHype* GetHypeCreator()
    { return (GateHype*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* HypeInnerRCmd;
  G4UIcmdWithADoubleAndUnit* HypeOuterRCmd;
  G4UIcmdWithADoubleAndUnit* HypeInnerStereoCmd;
  G4UIcmdWithADoubleAndUnit* HypeOuterStereoCmd;
  G4UIcmdWithADoubleAndUnit* HypeZLengthCmd;
};

#endif /* GATEHYPEMESSENGER_HH_ */
