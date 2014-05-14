/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateTransferEfficiencyMessenger_h
#define GateTransferEfficiencyMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "G4UIdirectory.hh"
#include "GateTransferEfficiency.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateTransferEfficiency;

class GateTransferEfficiencyMessenger: public GatePulseProcessorMessenger
{
public:
  GateTransferEfficiencyMessenger(GateTransferEfficiency* itsTE);
  virtual ~GateTransferEfficiencyMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
  inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

  inline GateTransferEfficiency* GetTransferEfficiency()
  { return (GateTransferEfficiency*) GetPulseProcessor(); }

private:
  G4UIcmdWithAString   *newVolCmd;
  std::vector<G4UIdirectory*> m_volDirectory;
  std::vector<G4UIcmdWithADouble*>   coeffTECmd;
  std::vector<G4String> m_name;
  G4int m_count;
};

#endif
