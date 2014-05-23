/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateObjectMoveMessenger_h
#define GateObjectMoveMessenger_h 1

#include "globals.hh"


#include "GateObjectRepeaterMessenger.hh"

class GateVGlobalPlacement;

/*! \class GateObjectMoveMessenger
    \brief Base class for all object-move messengers
    
    - GateObjectMoveMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateObjectMoveMessenger inherits from the abilities/responsabilities
      of the GateObjectRepeaterMessenger base-class: creation and management
      of a Gate UI directory for a Gate movement object; creation of the UI commands 
      "describe", "enable" and "disable"
      
    - In addition, it defines the method GetMove() that can return the
      object-move controled by the messenger

*/      
class GateObjectMoveMessenger: public GateObjectRepeaterMessenger
{
  public:
    //! constructor
    GateObjectMoveMessenger(GateVGlobalPlacement* itsMove);
    //! destructor
   ~GateObjectMoveMessenger() {}
    
    //! Returns the move controled by the messenger
    virtual inline GateVGlobalPlacement* GetMove() 
      { return (GateVGlobalPlacement*) GetObjectRepeater(); }
    
};

#endif

