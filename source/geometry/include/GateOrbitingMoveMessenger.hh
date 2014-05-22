/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateOrbitingMoveMessenger_h
#define GateOrbitingMoveMessenger_h 1

#include "globals.hh"

#include "GateObjectMoveMessenger.hh"
#include "GateOrbitingMove.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;


/*! \class GateOrbitingMoveMessenger
    \brief A messenger for a GateOrbitingMove (an orbiting)
    
    - GateOrbitingMoveMessenger - by Daniel.Strul@iphe.unil.ch
    
    - The GateOrbitingMoveMessenger inherits from the abilities/responsabilities
      of the GateObjectMoveMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it proposes and manages UI commands that are specific
      to an orbiting movement: 'setPoint1', 'setPoint2',
      'setVelocity', 'enableAutoRotation','disableAutoRotation'

*/      
class GateOrbitingMoveMessenger: public GateObjectMoveMessenger
{
  public:
    //! constructor
    GateOrbitingMoveMessenger(GateOrbitingMove* itsMove);
    //! destructor
   ~GateOrbitingMoveMessenger();
    
    //! Command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the orbiting move controled by the messenger
    virtual inline GateOrbitingMove* GetOrbitingMove() 
      { return (GateOrbitingMove*)GetMove(); }
    
  private:
    //! \name command objects
    //@{
  G4UIcmdWith3VectorAndUnit*   Point1Cmd;   //!< Command to set the first point of the orbit axis
  G4UIcmdWith3VectorAndUnit*   Point2Cmd;   //!< Command to set the second point of the orbit axis 
  G4UIcmdWithADoubleAndUnit*   VelocityCmd;   //!< Command to set the revolution velocity
  G4UIcmdWithABool*            EnableAutoRotationCmd;  //!< Command to enable the auto-rotation of the volume
  G4UIcmdWithABool*            DisableAutoRotationCmd; //!< Command to disable the auto-rotation of the volume
  //@}

};

#endif

