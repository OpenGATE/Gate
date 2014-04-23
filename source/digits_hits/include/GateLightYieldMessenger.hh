/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateLightYieldMessenger_h
#define GateLightYieldMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "GateLightYield.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithADouble;

class GateLightYield;

class GateLightYieldMessenger: public GatePulseProcessorMessenger
{
  public:
    GateLightYieldMessenger(GateLightYield* itsLightYield);
    virtual ~GateLightYieldMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GateLightYield* GetLightYield()
      { return (GateLightYield*) GetPulseProcessor(); }

  private:
    G4UIcmdWithAString   *newVolCmd;
    std::vector<G4UIdirectory*> m_volDirectory;
    std::vector<G4UIcmdWithADouble*>   lightOutputCmd;
    std::vector<G4String> m_name;
    G4int m_count;
};

#endif
