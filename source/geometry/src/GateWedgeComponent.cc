/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
  
#include "GateWedgeComponent.hh"
#include "GateWedge.hh"

GateWedgeComponent::GateWedgeComponent(const G4String& itsName,
      	      	      	      	      	      	 GateSystemComponent* itsMotherComponent,
		      	      	      	      	 GateVSystem* itsSystem)
  : GateSystemComponent(itsName,itsMotherComponent,itsSystem)
{
}




// Destructor
GateWedgeComponent::~GateWedgeComponent() 
{
}




// Method overloading the method IsValidAttachmentRequest() of the base-class GateSystemComponent
// It tells whether an creator may be attached to this component
// In addition to the test performed by the base-class' method, it also
// checks that the creator is indeed connected to a Wedge creator
G4bool GateWedgeComponent::IsValidAttachmentRequest(GateVVolume* anCreator) const
{
  // Call the base-class method to do all the standard validity tests
  if (!(GateSystemComponent::IsValidAttachmentRequest(anCreator)))
    return false;

  // Check whether the creator is connected to a Wedge-creator
  GateWedge* WedgeCreator = GetWedgeCreator(anCreator);
  if (!WedgeCreator) {
       G4cerr   << "[" << anCreator->GetObjectName() << "::IsValidAttachmentRequest]:" << G4endl
                << "\tThe volume creator ('" << WedgeCreator->GetObjectName() << "') for this creator does not seem to be a Wedge" << G4endl << G4endl;
      return false;
  }

  // OK, everything's fine
  return true;
}
   


// Return the length along an axis of the Wedge-creator attached to our creator
G4double GateWedgeComponent::GetWedgeLength(size_t axis) const
{
  return GetWedgeCreator() ? GetWedgeCreator()->GetWedgeLength(axis) : 0. ;
}



// Tool method: try to cast a creator into a Wedge creator
GateWedge* GateWedgeComponent::CastToWedgeCreator(GateVVolume* creator)
{ 
  return dynamic_cast<GateWedge*>(creator); 
}


