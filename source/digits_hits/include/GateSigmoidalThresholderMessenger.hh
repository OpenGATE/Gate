/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSigmoidalThresholderMessenger_h
#define GateSigmoidalThresholderMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include "GateSigmoidalThresholder.hh"

class G4UIdirectory;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;

class GateSigmoidalThresholder;

class GateSigmoidalThresholderMessenger: public GatePulseProcessorMessenger
{
  public:
    GateSigmoidalThresholderMessenger(GateSigmoidalThresholder* itsSigThresholder);
    virtual ~GateSigmoidalThresholderMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateSigmoidalThresholder* GetSigmoidalThresholder()
      { return (GateSigmoidalThresholder*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADoubleAndUnit  *thresholdCmd;
    G4UIcmdWithADouble         *alphaCmd;
    G4UIcmdWithADouble         *perCentCmd;
};

#endif
