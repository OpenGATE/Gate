#ifndef GATESOURCEY90BREM_MESSENGER_HH
#define GATESOURCEY90BREM_MESSENGER_HH

#include "GateVSourceMessenger.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

class GateSourceY90Brem;

class GateSourceY90BremMessenger : public GateVSourceMessenger
{
public:
  GateSourceY90BremMessenger(GateSourceY90Brem *source);
  ~GateSourceY90BremMessenger();

  void SetNewValue(G4UIcommand *, G4String);

protected:
  GateSourceY90Brem *mSource;

  G4UIcmdWithADoubleAndUnit* setMinBremEnergyCmd;
  G4UIcmdWithADouble* setPosProbabilityCmd;
};

#endif
