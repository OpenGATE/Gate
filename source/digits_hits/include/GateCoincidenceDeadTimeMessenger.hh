/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateCoincidenceDeadTimeMessenger
    \brief  Messenger for the GateCoincidenceDeadTime

    - GateCoincidenceDeadTime - by unknown author

    \sa GateCoincidenceDeadTime, GateCoincidenceDeadTimeMessenger
*/


#ifndef GateCoincidenceDeadTimeMessenger_h
#define GateCoincidenceDeadTimeMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateCoincidenceDeadTime;
class G4UIcmdWithAString;

class GateCoincidenceDeadTimeMessenger : public GateClockDependentMessenger
{
public:
  
  GateCoincidenceDeadTimeMessenger(GateCoincidenceDeadTime*);
  ~GateCoincidenceDeadTimeMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateCoincidenceDeadTime* m_CoincidenceDeadTime;
  G4UIcmdWithADoubleAndUnit *deadTimeCmd; //!< set the dead time value
  G4UIcmdWithAString   *modeCmd;          //!< set the dead time mode
  G4UIcmdWithADoubleAndUnit   *bufferSizeCmd; //!< set the buffer size
  G4UIcmdWithAnInteger   *bufferModeCmd;      //!< set the buffer usage mode
  G4UIcmdWithABool   *conserveAllEventCmd;    //!< Tell if an event is entierly ok or not



};

#endif








