/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateVolumePlacementMessenger_h
#define GateVolumePlacementMessenger_h 1

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

class GateVolumePlacement;

/*! \class GateVolumePlacementMessenger
    \brief A messenger for a GateVolumePlacement (a static placement)
    
    - GateVolumePlacementMessenger - by Daniel.Strul@iphe.unil.ch
    
    - The GateVolumePlacementMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it proposes and manages UI commands that are specific
      to a static placement: 'setTranslation', 'setRotationAngle',
      'setRotationAxis', 'alignToX', 'alignToY', 'alignToZ'

*/      
//--------------------------------------------------------------------
class GateVolumePlacementMessenger: public GateObjectRepeaterMessenger
{
  public:
    //! constructor
    GateVolumePlacementMessenger(GateVolumePlacement* itsPlacementMove);
    //! destructor
   ~GateVolumePlacementMessenger();
    
    //! Command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    //! Returns the placement move controled by the messenger
    virtual inline GateVolumePlacement* GetVolumePlacement() 
      { return (GateVolumePlacement*)GetObjectRepeater(); }
    
  private:
    //! \name command objects
    //@{
    G4UIcmdWith3VectorAndUnit*  TranslationCmd;   //!< Command to set the translation vector
    G4UIcmdWithADoubleAndUnit*  RotationAngleCmd; //!< Command to set the rotation angle
    G4UIcmdWith3Vector*         RotationAxisCmd;  //!< Command to set the rotation axis
    G4UIcmdWithoutParameter*    AlignToXCmd;  	  //!< Command to align the object with the X axis
    G4UIcmdWithoutParameter*    AlignToYCmd;  	  //!< Command to align the object with the Y axis
    G4UIcmdWithoutParameter*    AlignToZCmd;  	  //!< Command to align the object with the Z axis
    G4UIcmdWithADoubleAndUnit*  SetPhiCmd;    	  //!< Command to set the phi of translation vector
    G4UIcmdWithADoubleAndUnit*  SetThetaCmd;      //!< Command to set the theta of translation vector
    G4UIcmdWithADoubleAndUnit*  SetMagCmd;    	  //!< Command to set the mag of translation vector
    //@}

};
//--------------------------------------------------------------------

#endif

