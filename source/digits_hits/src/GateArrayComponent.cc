/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateArrayComponent.hh"

#include "GateArrayRepeater.hh"
#include "GateObjectRepeaterList.hh"


/* Constructor

    itsName:       	  the name chosen for this system-component
    itsMotherComponent:   the mother of the component (0 if top of a tree)
    itsSystem:            the system to which the component belongs
*/
GateArrayComponent::GateArrayComponent(const G4String& itsName,
      	      	      	      	       GateSystemComponent* itsMotherComponent,
		      	      	       GateVSystem* itsSystem)
  : GateBoxComponent( itsName , itsMotherComponent,itsSystem)
{
}




// Destructor
GateArrayComponent::~GateArrayComponent()
{
}




// Finds the first array-repeater in the inserter's repeater list
GateArrayRepeater* GateArrayComponent::FindArrayRepeater()
{
  return FindRepeater<GateArrayRepeater>();
}





// Finds the first array-repeater's repeat number along an axis
G4int GateArrayComponent::GetRepeatNumber(size_t axis)
{
  // Get the inserter's first array-repeater
  GateArrayRepeater* repeater = FindArrayRepeater();
  // Return the repeater's repeat-number along the axis
  return repeater ? repeater->GetRepeatNumber(axis) : 1;
}



// Finds the first array-repeater's repeat vector
const G4ThreeVector& GateArrayComponent::GetRepeatVector()
{
  static const G4ThreeVector defaultRepeatVector;

  // Get the inserter's first array-repeater
  GateArrayRepeater* repeater = FindArrayRepeater();
  // Return the repeater's repeat-vector
  return repeater ? repeater->GetRepeatVector() : defaultRepeatVector;
}
