/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCrystalBlurringMessenger_h
#define GateCrystalBlurringMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateCrystalBlurring;

class GateCrystalBlurringMessenger: public GatePulseProcessorMessenger
{
  public:
    GateCrystalBlurringMessenger(GateCrystalBlurring* itsCrystalresolution);
    virtual ~GateCrystalBlurringMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateCrystalBlurring* GetCrystalBlurring()
      { return (GateCrystalBlurring*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADouble   *crystalresolutionminCmd;
    G4UIcmdWithADouble   *crystalresolutionmaxCmd;
    G4UIcmdWithADouble   *crystalQECmd;
    G4UIcmdWithADoubleAndUnit   *crystalerefCmd;
};

#endif
