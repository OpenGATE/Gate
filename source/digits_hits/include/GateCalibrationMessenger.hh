/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCalibrationMessenger_h
#define GateCalibrationMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithADouble;

class GateCalibration;

class GateCalibrationMessenger: public GatePulseProcessorMessenger
{
  public:
    GateCalibrationMessenger(GateCalibration* itsCalibration);
    virtual ~GateCalibrationMessenger() {};

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateCalibration* GetCalibration()
      { return (GateCalibration*) GetPulseProcessor(); }

private:
  G4UIcmdWithADouble   *calibCmd;

};

#endif
