/*
 * GateEllipticalConeMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEELLIPTICALCONEMESSENGER_HH_
#define GATEELLIPTICALCONEMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateEllipticalCone;

class GateEllipticalConeMessenger: public GateVolumeMessenger
{
public:
  GateEllipticalConeMessenger(GateEllipticalCone* itsCreator);
  ~GateEllipticalConeMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateEllipticalCone* GetEllipticalConeCreator()
    { return (GateEllipticalCone*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* EllipticalConeXSemiAxisCmd;
  G4UIcmdWithADoubleAndUnit* EllipticalConeYSemiAxisCmd;
  G4UIcmdWithADoubleAndUnit* EllipticalConeZLengthCmd;
  G4UIcmdWithADoubleAndUnit* EllipticalConeZCutCmd;
};

#endif /* GATEELLIPTICALCONEMESSENGER_HH_ */
