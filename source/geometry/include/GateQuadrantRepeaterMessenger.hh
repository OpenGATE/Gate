/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateQuadrantRepeaterMessenger_h
#define GateQuadrantRepeaterMessenger_h 1

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

class GateQuadrantRepeater;




/*! \class GateQuadrantRepeaterMessenger
    \brief Messenger for a GateQuadrantRepeater
    
    - GateQuadrantRepeaterMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateQuadrantRepeaterMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; UI commands "describe",
      'enable' and 'disable'.
      
    - In addition, it creates UI commands to manage a quadrant repeater:
      'setLineNumber', 'setOrientation', 'setCopySpacing', 'setMaxRange'

*/      
class GateQuadrantRepeaterMessenger: public GateObjectRepeaterMessenger
{
  public:
    GateQuadrantRepeaterMessenger(GateQuadrantRepeater* itsQuadrantRepeater);
   ~GateQuadrantRepeaterMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  public:
    virtual inline GateQuadrantRepeater* GetQuadrantRepeater() 
      { return (GateQuadrantRepeater*)GetObjectRepeater(); }
    
  private:
    G4UIcmdWithAnInteger*      SetLineNumberCmd;
    G4UIcmdWithADoubleAndUnit* SetOrientationCmd;
    G4UIcmdWithADoubleAndUnit* SetCopySpacingCmd;
    G4UIcmdWithADoubleAndUnit* SetMaxRangeCmd;

};

#endif

