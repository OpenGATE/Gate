/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTranslationMoveMessenger_h
#define GateTranslationMoveMessenger_h 1

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

class GateTranslationMove;

/*! \class GateTranslationMoveMessenger
    \brief A messenger for a GateTranslationMove (a constant speed translation)
    
    - GateTranslationMoveMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateTranslationMoveMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it proposes and manages UI commands that are specific
      to an oscillating translation movement: 'setVelocity'

*/      
class GateTranslationMoveMessenger: public GateObjectRepeaterMessenger
{
  public:
    //! constructor
    GateTranslationMoveMessenger(GateTranslationMove* itsTranslationMove);
    //! destructor
   ~GateTranslationMoveMessenger();
    
    //! Command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the translation move controled by the messenger
    virtual inline GateTranslationMove* GetTranslationMove() 
      { return (GateTranslationMove*)GetObjectRepeater(); }
    
  private:
    //! \name command objects
    //@{
    G4UIcmdWith3VectorAndUnit* TranslationVelocityCmd; //!< Command to set the velocity vector
    //@}

};

#endif

