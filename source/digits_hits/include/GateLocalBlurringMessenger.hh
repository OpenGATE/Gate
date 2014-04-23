/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateLocalBlurringMessenger_h
#define GateLocalBlurringMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "G4UIdirectory.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateLocalBlurring;

class GateLocalBlurringMessenger: public GatePulseProcessorMessenger
{
public:
  GateLocalBlurringMessenger(GateLocalBlurring* itsResolution);
  virtual ~GateLocalBlurringMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
  inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

  inline GateLocalBlurring* GetLocalBlurring()
  { return (GateLocalBlurring*) GetPulseProcessor(); }

private:
  G4UIcmdWithAString   *newVolCmd;
  std::vector<G4UIdirectory*> m_volDirectory;
  std::vector<G4UIcmdWithADouble*>   resolutionCmd;
  std::vector<G4UIcmdWithADoubleAndUnit*>   erefCmd;
  std::vector<G4String> m_name;
  G4int m_count;
};

#endif
