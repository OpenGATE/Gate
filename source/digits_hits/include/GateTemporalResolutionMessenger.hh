/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTemporalResolutionMessenger_h
#define GateTemporalResolutionMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIcmdWithADoubleAndUnit;

class GateTemporalResolution;

class GateTemporalResolutionMessenger: public GatePulseProcessorMessenger
{
  public:
    GateTemporalResolutionMessenger(GateTemporalResolution* itsTemporalResolution);
    virtual ~GateTemporalResolutionMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateTemporalResolution* GetTemporalResolution()
      { return (GateTemporalResolution*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADoubleAndUnit   *timeResolutionCmd;
};

#endif
