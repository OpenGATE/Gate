/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateOscTranslationMoveMessenger_h
#define GateOscTranslationMoveMessenger_h 1

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

class GateOscTranslationMove;


/*! \class GateOscTranslationMoveMessenger
    \brief A messenger for a GateOscTranslationMove (an oscillating translation)
    
    - GateOscTranslationMoveMessenger - by Daniel.Strul@iphe.unil.ch (Aug. 10, 2002)
    
    - The GateOscTranslationMoveMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it proposes and manages UI commands that are specific
      to an oscillating translation movement: 'setAmplitude', 'setFrequency',
      'setPeriod', 'setPhase'

*/      
class GateOscTranslationMoveMessenger: public GateObjectRepeaterMessenger
{
  public:
    //! constructor
    GateOscTranslationMoveMessenger(GateOscTranslationMove* itsTranslationMove);
    //! destructor
   ~GateOscTranslationMoveMessenger();
    
    //! Command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the oscillating translation move controled by the messenger
    virtual inline GateOscTranslationMove* GetTranslationMove() 
      { return (GateOscTranslationMove*)GetObjectRepeater(); }
    
  private:
    //! \name command objects
    //@{
    G4UIcmdWith3VectorAndUnit*  AmplitudeCmd; 	//!< Command to set the maximum displacement vector
    G4UIcmdWithADoubleAndUnit*  FrequencyCmd; 	//!< Command to set the oscillation frequency
    G4UIcmdWithADoubleAndUnit*  PeriodCmd;    	//!< Command to set the oscillation period
    G4UIcmdWithADoubleAndUnit*  PhaseCmd;     	//!< Command to set the oscillation phase a t=0
    //@}
};

#endif

