/*----------------------
   OpenGATE Collaboration 
     
   Daniel Strul <daniel.strul@iphe.unil.ch> 
     
   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne 

This software is distributed under the terms 
of the GNU Lesser General  Public Licence (LGPL) 
See GATE/LICENSE.txt for further details 
----------------------*/

/*!

  \file GateCylindricalPETSystemMessenger.hh
  
  $Log: GateCylindricalPETSystemMessenger.hh,v $
  Revision 1.9  2003/07/02 22:19:44  dstrul
  Digitizer re-engineering

  Revision 1.8  2002/10/07 19:15:48  dstrul
  Added a bit of Doxygen/CVS documentation

  Revision 1.7  2002/08/11 15:33:24  dstrul
  Cosmetic cleanup: standardized file comments for cleaner doxygen output

  Revision 1.6  2002/07/21 19:34:50  dstrul
  Modified the way type-names are generated and handled

  \brief Class GateCylindricalPETSystemMessenger
  \brief By Daniel.Strul@iphe.unil.ch
  \brief $Id: GateCylindricalPETSystemMessenger.hh,v 1.9 2003/07/02 22:19:44 dstrul Exp $
*/

#ifndef GateCylindricalPETSystemMessenger_h
#define GateCylindricalPETSystemMessenger_h 1

#include "GateClockDependentMessenger.hh"

class GateCylindricalPETSystem;
class G4UIcmdWithAString;


/*! \class GateCylindricalPETSystemMessenger
    \brief Base class for GateCylindricalPETSystem messengers
    
    - GateCylindricalPETSystemMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateCylindricalPETSystemMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; creation of the UI command "describe"
      
    - In addition, it proposes and manages the UI commands 'enable' and 'disable'.

*/      
class GateCylindricalPETSystemMessenger: public GateClockDependentMessenger
{
  public:
    //! Constructor
    //! The flags are passed to the base-class GateClockDependentMessenger
    GateCylindricalPETSystemMessenger(GateCylindricalPETSystem* itsCylindricalPETSystem,
    			        const G4String& itsDirectoryName="");

   ~GateCylindricalPETSystemMessenger();  //!< destructor
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Get the clock-dependent object
    inline GateCylindricalPETSystem* GetCylindricalPETSystem() 
      { return (GateCylindricalPETSystem*) GetClockDependent(); }

  private:

    G4UIcmdWithAString*    addNewRsectorcmd;
};

#endif

