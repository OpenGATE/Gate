#ifndef GATESOURCEYFAST90_MESSENGER_HH
#define GATESOURCEYFAST90_MESSENGER_HH

#include "GateVSourceMessenger.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

class GateSourceFastY90;

class GateSourceFastY90Messenger : public GateVSourceMessenger
{
public:
  GateSourceFastY90Messenger(GateSourceFastY90 *source);
  ~GateSourceFastY90Messenger();

  void SetNewValue(G4UIcommand *, G4String);

protected:
  GateSourceFastY90 *mSource;

  G4UIcmdWithADoubleAndUnit* setMinBremEnergyCmd;
  G4UIcmdWithADouble* setPosProbabilityCmd;
  G4UIcmdWithAString* loadVoxelizedPhantomCmd;
  G4UIcmdWith3VectorAndUnit* setPhantomPositionCmd;
};

#endif
