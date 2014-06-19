/*
 * GateGenericTrapMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEGENERICTRAPMESSENGER_HH_
#define GATEGENERICTRAPMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateGenericTrap;

class GateGenericTrapMessenger: public GateVolumeMessenger
{
public:
  GateGenericTrapMessenger(GateGenericTrap* itsCreator);
  ~GateGenericTrapMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateGenericTrap* GetGenericTrapCreator()
    { return (GateGenericTrap*)GetVolumeCreator(); }

private:
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex1Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex2Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex3Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex4Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex5Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex6Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex7Cmd;
  G4UIcmdWith3VectorAndUnit* GenericTrapVertex8Cmd;
  G4UIcmdWithADoubleAndUnit* GenericTrapZLengthCmd;
};

#endif /* GATEGENERICTRAPMESSENGER_HH_ */
