/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateArrayComponent_h
#define GateArrayComponent_h 1

#include "GateBoxComponent.hh"

class GateArrayRepeater;

/*! \class  GateArrayComponent
    \brief  A GateArrayComponent is a subtype of GateBoxComponent
    \brief  It is meant to be connected to box inserters that are repeated
    \brief in arrays, such as crystal matrices

    - GateArrayComponent - by Daniel.Strul@iphe.unil.ch (Oct 2002)

    - A GateArrayComponent is derived from GateBoxComponent, and thus inherits all
      the properties of boxcreator-components. In addition, it includes some methods
      that are specific to inserters repeated by array-repeaters: FindArrayRepeater(),
      GetRepeatNumber() and GetRepeatVector()

    \sa GateSystemComponent, GateBoxComponent, GateArrayRepeater
*/
class GateArrayComponent  : public GateBoxComponent
{
  public:
    /*! \brief Constructor

        \param itsName:             the name chosen for this system-component
        \param itsMotherComponent:  the mother of the component (0 if top of a tree)
        \param itsSystem:           the system to which the component belongs
    */
    GateArrayComponent(const G4String& itsName,
                       GateSystemComponent* itsMotherComponent,
                       GateVSystem* itsSystem);
    //! Destructor
    virtual ~GateArrayComponent();

    //! Finds the first array-repeater in the inserter's repeater list
    GateArrayRepeater* FindArrayRepeater();

    //! Finds the first array-repeater's repeat number along an axis
    G4int GetRepeatNumber(size_t axis);

    //! Finds the first array-repeater's repeat vector
    const G4ThreeVector& GetRepeatVector();

};

#endif
