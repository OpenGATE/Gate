/*
 * GateParaMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATEPARAMESSENGER_HH_
#define GATEPARAMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GatePara;

class GateParaMessenger: public GateVolumeMessenger
{
public:
  GateParaMessenger(GatePara* itsCreator);
  ~GateParaMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GatePara* GetParaCreator()
    { return (GatePara*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* ParaXLengthCmd;
  G4UIcmdWithADoubleAndUnit* ParaYLengthCmd;
  G4UIcmdWithADoubleAndUnit* ParaZLengthCmd;
  G4UIcmdWithADoubleAndUnit* ParaAlphaCmd;
  G4UIcmdWithADoubleAndUnit* ParaThetaCmd;
  G4UIcmdWithADoubleAndUnit* ParaPhiCmd;
};

#endif /* GATEPARAMESSENGER_HH_ */
