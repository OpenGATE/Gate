/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateLinearRepeaterMessenger_h
#define GateLinearRepeaterMessenger_h 1

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

class GateLinearRepeater;

/*! \class GateLinearRepeaterMessenger
    \brief Messenger for a GateLinearRepeater
    
    - GateLinearRepeaterMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateLinearRepeaterMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; UI commands "describe",
      'enable' and 'disable'.
      
    - In addition, it creates UI commands to manage a linear repeater:
      'setRepeatVector', 'setRepeatNumber', 'autoCenter'

*/      

class GateLinearRepeaterMessenger: public GateObjectRepeaterMessenger
{
  public:
    GateLinearRepeaterMessenger(GateLinearRepeater* itsLinearRepeater);
   ~GateLinearRepeaterMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  public:
    virtual inline GateLinearRepeater* GetLinearRepeater() 
      { return (GateLinearRepeater*)GetObjectRepeater(); }
    
  private:
    G4UIcmdWith3VectorAndUnit* SetRepeatVectorCmd;
    G4UIcmdWithAnInteger*      SetRepeatNumberCmd;
    G4UIcmdWithABool* 	       AutoCenterCmd;

};

#endif

