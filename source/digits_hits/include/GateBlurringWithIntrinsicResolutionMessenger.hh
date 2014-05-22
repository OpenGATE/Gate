/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateBlurringWithIntrinsicResolutionMessenger_h
#define GateBlurringWithIntrinsicResolutionMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "G4UIdirectory.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;

class GateBlurringWithIntrinsicResolution;

class GateBlurringWithIntrinsicResolutionMessenger: public GatePulseProcessorMessenger
{
public:
  GateBlurringWithIntrinsicResolutionMessenger(GateBlurringWithIntrinsicResolution* itsIntrinsic);
  virtual ~GateBlurringWithIntrinsicResolutionMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
  inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

  inline GateBlurringWithIntrinsicResolution* GetBlurringWithIntrinsicResolution()
  { return (GateBlurringWithIntrinsicResolution*) GetPulseProcessor(); }

private:
  G4UIcmdWithAString   *newVolCmd;
  std::vector<G4UIdirectory*> m_volDirectory;
  std::vector<G4UIcmdWithADouble*>   resolutionCmd;
  std::vector<G4UIcmdWithADoubleAndUnit*>   erefCmd;
  std::vector<G4String> m_name;
  G4int m_count;
};

#endif
