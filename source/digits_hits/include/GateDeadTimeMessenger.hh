/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDeadTimeMessenger_h
#define GateDeadTimeMessenger_h 1

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

class GateDeadTime;

class GateDeadTimeMessenger: public GatePulseProcessorMessenger
{
public:
  GateDeadTimeMessenger(GateDeadTime* itsDeadTime);
  virtual ~GateDeadTimeMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  inline GateDeadTime* GetDeadTime(){ return (GateDeadTime*) GetPulseProcessor(); }

private:
  G4UIcmdWithADoubleAndUnit *deadTimeCmd; //!< set the dead time value
  G4UIcmdWithAString   *newVolCmd;        //!< set the geometric level of application
  G4UIcmdWithAString   *modeCmd;          //!< set the dead time mode
  G4UIcmdWithADoubleAndUnit   *bufferSizeCmd; //!< set the buffer size
  G4UIcmdWithAnInteger   *bufferModeCmd;      //!< set the buffer usage mode
};

#endif
