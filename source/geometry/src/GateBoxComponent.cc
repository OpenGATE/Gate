/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBoxComponent.hh"
#include "GateBox.hh"

/* Constructor

    itsName:       	  the name chosen for this system-component
    itsMotherComponent:   the mother of the component (0 if top of a tree)
    itsSystem:            the system to which the component belongs
*/    
GateBoxComponent::GateBoxComponent(const G4String& itsName,
      	      	      	      	      	      	 GateSystemComponent* itsMotherComponent,
		      	      	      	      	 GateVSystem* itsSystem)
  : GateSystemComponent(itsName,itsMotherComponent,itsSystem)
{
}




// Destructor
GateBoxComponent::~GateBoxComponent() 
{
}




// Method overloading the method IsValidAttachmentRequest() of the base-class GateSystemComponent
// It tells whether an inserter may be attached to this component
// In addition to the test performed by the base-class' method, it also
// checks that the inserter is indeed connected to a box creator
G4bool GateBoxComponent::IsValidAttachmentRequest(GateVVolume* anCreator) const
{
  // Call the base-class method to do all the standard validity tests
  if (!(GateSystemComponent::IsValidAttachmentRequest(anCreator)))
    return false;

  // Check whether the inserter is connected to a box-creator
  GateBox* boxCreator = GetBoxCreator(anCreator);
  if (!boxCreator) {
       G4cerr   << "[" << anCreator->GetObjectName() << "::IsValidAttachmentRequest]:" << G4endl
                << "\tThe volume creator ('" << boxCreator->GetObjectName() << "') for this inserter does not seem to be a box" << G4endl << G4endl;
      return false;
  }

  // OK, everything's fine
  return true;
}
   


// Return the length along an axis of the box-creator attached to our inserter
G4double GateBoxComponent::GetBoxLength(size_t axis) const
{
  return GetBoxCreator() ? GetBoxCreator()->GetBoxLength(axis) : 0. ;
}



// Tool method: try to cast a creator into a box creator
GateBox* GateBoxComponent::CastToBoxCreator(GateVVolume* creator)
{ 
  return dynamic_cast<GateBox*>(creator); 
}


