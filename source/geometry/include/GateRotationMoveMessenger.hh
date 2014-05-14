/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateRotationMoveMessenger_h
#define GateRotationMoveMessenger_h 1

#include "globals.hh"

#include "GateObjectRepeaterMessenger.hh"
#include "GateRotationMove.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;


/*! \class GateRotationMoveMessenger
    \brief A messenger for a GateRotationMove (a constant speed rotation)
    
    - GateRotationMoveMessenger - by Daniel.Strul@iphe.unil.ch
    
    - The GateRotationMoveMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it proposes and manages UI commands that are specific
      to a rotation movement: 'setRotationAxis', 'setVelocity'

*/      
class GateRotationMoveMessenger: public GateObjectRepeaterMessenger
{
  public:
    //! constructor
    GateRotationMoveMessenger(GateRotationMove* itsMove);
    //! destructor
   ~GateRotationMoveMessenger();
    
    //! Command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the rotation move controled by the messenger
    virtual inline GateRotationMove* GetRotationMove() 
      { return (GateRotationMove*)GetObjectRepeater(); }
    
  private:
    //! \name command objects
    //@{
    G4UIcmdWith3Vector*        RotationAxisCmd; //!< Command to set the rotation axis
    G4UIcmdWithADoubleAndUnit* VelocityCmd;   	//!< Command to set the angular velocity
    //@}


};

#endif

