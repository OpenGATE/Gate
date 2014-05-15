/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateArrayRepeaterMessenger_h
#define GateArrayRepeaterMessenger_h 1

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

class GateArrayRepeater;

/*! \class GateArrayRepeaterMessenger
    \brief Messenger for a GateArrayRepeater
    
    - GateArrayRepeaterMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateArrayRepeaterMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; UI commands "describe",
      'enable' and 'disable'.
      
    - In addition, it creates UI commands to manage a matrix repeater:
      'setRepeatVector', 'setRepeatNumberX', 'setRepeatNumberY', 
      'setRepeatNumberZ', 'autoCenter'

*/      
class GateArrayRepeaterMessenger: public GateObjectRepeaterMessenger
{
  public:
    GateArrayRepeaterMessenger(GateArrayRepeater* itsCubicArrayRepeater);
   ~GateArrayRepeaterMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  public:
    virtual inline GateArrayRepeater* GetCubicArrayRepeater() 
      { return (GateArrayRepeater*)GetObjectRepeater(); }
    
  private:
    G4UIcmdWithAnInteger*      SetRepeatNumberXCmd;
    G4UIcmdWithAnInteger*      SetRepeatNumberYCmd;
    G4UIcmdWithAnInteger*      SetRepeatNumberZCmd;
    G4UIcmdWith3VectorAndUnit* SetRepeatVectorCmd;
    G4UIcmdWithABool* 	       AutoCenterCmd;

};

#endif

