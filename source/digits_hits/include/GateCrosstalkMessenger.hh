/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCrosstalkMessenger_h
#define GateCrosstalkMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "G4UIdirectory.hh"

class G4UIcmdWithAString;
class G4UIcmdWithADouble;

class GateCrosstalk;

class GateCrosstalkMessenger: public GatePulseProcessorMessenger
{
  public:
    GateCrosstalkMessenger(GateCrosstalk* itsCrosstalk);
    virtual ~GateCrosstalkMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateCrosstalk* GetCrosstalk()
      { return (GateCrosstalk*) GetPulseProcessor(); }

  private:
    G4UIcmdWithAString   *newVolCmd;
    G4UIcmdWithADouble   *edgesFractionCmd;
    G4UIcmdWithADouble   *cornersFractionCmd;
};

#endif
