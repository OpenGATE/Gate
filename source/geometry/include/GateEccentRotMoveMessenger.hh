/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateEccentRotMoveMessenger_h
#define GateEccentRotMoveMessenger_h 1

#include "globals.hh"

#include "GateObjectRepeaterMessenger.hh"
#include "GateEccentRotMove.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;


/*! \class GateEccentRotMoveMessenger
    \brief A messenger for a GateEccentRotMove (an orbiting)
    
    - GateEccentRotMoveMessenger - by Daniel.Strul@iphe.unil.ch
    
    - The GateEccentRotMoveMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it proposes and manages UI commands that are specific
      to an orbiting movement: 'setShift','setVelocity'


*/      
class GateEccentRotMoveMessenger: public GateObjectRepeaterMessenger
{
  public:
    //! constructor
    GateEccentRotMoveMessenger(GateEccentRotMove* itsMove);
    //! destructor
   ~GateEccentRotMoveMessenger();
    
    //! Command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the orbiting move controled by the messenger
    virtual inline GateEccentRotMove* GetEccentRotMove() 
      { return (GateEccentRotMove*)GetObjectRepeater(); }
    
  private:
    //! \name command objects
    //@{
    G4UIcmdWith3VectorAndUnit*        ShiftCmd;     //!< Command to set the first point of the orbit axis
    G4UIcmdWithADoubleAndUnit*        VelocityCmd;   //!< Command to set the revolution velocity
    //@}

};

#endif
