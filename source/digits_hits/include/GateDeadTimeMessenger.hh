/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GateDeadTimeMessenger
    \brief  Messenger for the GateDeadTime

    - GateDeadTime - by Luc.Simon@iphe.unil.ch

    \sa GateDeadTime, GateDeadTimeMessenger
*/


#ifndef GateDeadTimeMessenger_h
#define GateDeadTimeMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"

class GateDeadTime;

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateDeadTimeMessenger : public GateClockDependentMessenger
{
public:
  
  GateDeadTimeMessenger(GateDeadTime*);
  ~GateDeadTimeMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateDeadTime* m_DeadTime;

  G4UIcmdWithADoubleAndUnit *DeadTimeCmd; //!< set the dead time value
  G4UIcmdWithAString   *newVolCmd;        //!< set the geometric level of application
  G4UIcmdWithAString   *modeCmd;          //!< set the dead time mode
  G4UIcmdWithADoubleAndUnit   *bufferSizeCmd; //!< set the buffer size
  G4UIcmdWithAnInteger   *bufferModeCmd;      //!< set the buffer usage mode


};

#endif








