/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePileupMessenger_h
#define GatePileupMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;

class GatePileup;

class GatePileupMessenger: public GatePulseProcessorMessenger
{
  public:
    GatePileupMessenger(GatePileup* itsPileup);
    virtual~GatePileupMessenger();

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GatePileup* GetPileup()
      { return (GatePileup*) GetPulseProcessor(); }

  private:
    G4UIcmdWithAnInteger*      SetDepthCmd;
    G4UIcmdWithADoubleAndUnit* SetPileupCmd;
};

#endif
