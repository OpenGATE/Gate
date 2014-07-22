/*
 * GateTetraMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETETRAMESSENGER_HH_
#define GATETETRAMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTetra;

class GateTetraMessenger: public GateVolumeMessenger
{
public:
  GateTetraMessenger(GateTetra* itsCreator);
  ~GateTetraMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateTetra* GetTetraCreator()
    { return (GateTetra*)GetVolumeCreator(); }

private:
  G4UIcmdWith3VectorAndUnit* TetraP1Cmd;
  G4UIcmdWith3VectorAndUnit* TetraP2Cmd;
  G4UIcmdWith3VectorAndUnit* TetraP3Cmd;
  G4UIcmdWith3VectorAndUnit* TetraP4Cmd;
};



#endif /* GATETETRAMESSENGER_HH_ */
