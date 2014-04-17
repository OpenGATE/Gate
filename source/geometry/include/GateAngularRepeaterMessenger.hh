/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateAngularRepeaterMessenger_h
#define GateAngularRepeaterMessenger_h 1

#include "globals.hh"
#include "GateObjectRepeaterMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;

class GateAngularRepeater;

/*! \class GateAngularRepeaterMessenger
  \brief Messenger for a GateAngularRepeater

  - GateAngularRepeaterMessenger - by Daniel.Strul@iphe.unil.ch

  - The GateAngularRepeaterMessenger inherits from the abilities/responsabilities
  of the GateObjectRepeaterMessenger base-class: creation and management
  of a Gate UI directory for a Gate object; UI commands "describe",
  'enable' and 'disable'.

  - In addition, it creates UI commands to manage a matrix repeater:
  'setRepeatNumber', 'setPoint1', 'setPoint2', 'enableAutoRotation',
  'disableAutoRotation', 'setFirstAngle', 'setAngularSpan',
  'setWichNoShifted','setModuloNumber','setZShift1 .. 8'
  We can define up to 8 differents shifts for repeater.

*/
class GateAngularRepeaterMessenger: public GateObjectRepeaterMessenger
{
public:
  GateAngularRepeaterMessenger(GateAngularRepeater* itsAngularRepeater);
  ~GateAngularRepeaterMessenger();

  void SetNewValue(G4UIcommand*, G4String);

public:
  virtual inline GateAngularRepeater* GetAngularRepeater()
  { return (GateAngularRepeater*)GetObjectRepeater(); }

private:
  G4UIcmdWithAnInteger*       SetRepeatNumberCmd;
  G4UIcmdWith3Vector*         Point1Cmd;
  G4UIcmdWith3Vector*         Point2Cmd;
  G4UIcmdWithABool* 	        EnableAutoRotationCmd;
  G4UIcmdWithABool* 	        DisableAutoRotationCmd;
  G4UIcmdWithADoubleAndUnit*	FirstAngleCmd;
  G4UIcmdWithADoubleAndUnit*	AngularSpanCmd;
  G4UIcmdWithAnInteger*       SetModuloNumberCmd;
  G4UIcmdWithADoubleAndUnit*  Shift1Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift2Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift3Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift4Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift5Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift6Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift7Cmd;
  G4UIcmdWithADoubleAndUnit*  Shift8Cmd;
};

#endif
