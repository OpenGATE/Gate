/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceDeadTimeMessenger_h
#define GateCoincidenceDeadTimeMessenger_h 1

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

class GateCoincidenceDeadTime;

class GateCoincidenceDeadTimeMessenger: public GateClockDependentMessenger
{
public:
  GateCoincidenceDeadTimeMessenger(GateCoincidenceDeadTime* itsDeadTime);
  virtual ~GateCoincidenceDeadTimeMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  inline GateCoincidenceDeadTime* GetDeadTime(){ return (GateCoincidenceDeadTime*) GetClockDependent(); }

private:
  G4UIcmdWithADoubleAndUnit *deadTimeCmd; //!< set the dead time value
  G4UIcmdWithAString   *modeCmd;          //!< set the dead time mode
  G4UIcmdWithADoubleAndUnit   *bufferSizeCmd; //!< set the buffer size
  G4UIcmdWithAnInteger   *bufferModeCmd;      //!< set the buffer usage mode
  G4UIcmdWithABool   *conserveAllEventCmd;    //!< Tell if an event is entierly ok or not
};

#endif
