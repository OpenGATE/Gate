/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCylinderComponent.hh"
#include "GateCylinder.hh"

/* Constructor

    itsName:       	  the name chosen for this system-component
    itsMotherComponent:   the mother of the component (0 if top of a tree)
    itsSystem:            the system to which the component belongs
*/    
GateCylinderComponent::GateCylinderComponent(const G4String& itsName,
      	      	      	      	      	      	 GateSystemComponent* itsMotherComponent,
		      	      	      	      	 GateVSystem* itsSystem)
  : GateSystemComponent(itsName,itsMotherComponent,itsSystem)
{
}




// Destructor
GateCylinderComponent::~GateCylinderComponent() 
{
}

// Method overloading the method IsValidAttachmentRequest() of the base-class GateSystemComponent
// It tells whether an inserter may be attached to this component
// In addition to the test performed by the base-class' method, it also
// checks that the inserter is indeed connected to a cylinder creator
G4bool GateCylinderComponent::IsValidAttachmentRequest(GateVVolume* anCreator) const
{
  // Call the base-class method to do all the standard validity tests
  if (!(GateSystemComponent::IsValidAttachmentRequest(anCreator)))
    return false;
  // Check whether the inserter is connected to a cylinder-creator
  GateCylinder* cylinderCreator = GetCylinderCreator(anCreator);
  if (!cylinderCreator) {
       G4cerr   << "[" << anCreator->GetObjectName() << "::IsValidAttachmentRequest]:" << G4endl
                << "\tThe volume creator ('" << cylinderCreator->GetObjectName() << "') for this inserter does not seem to be a cylinder" << G4endl << G4endl;
      return false;
  }

  // OK, everything's fine
  return true;
}
   

G4double GateCylinderComponent::GetCylinderHeight() const
{
  return GetCylinderCreator() ? GetCylinderCreator()->GetCylinderHeight() : 0. ;
}
G4double GateCylinderComponent::GetCylinderRmin() const
{
  return GetCylinderCreator() ? GetCylinderCreator()->GetCylinderRmin() : 0. ;
}
G4double GateCylinderComponent::GetCylinderRmax() const
{
  return GetCylinderCreator() ? GetCylinderCreator()->GetCylinderRmax() : 0. ;
}
G4double GateCylinderComponent::GetCylinderSPhi() const
{
  return GetCylinderCreator() ? GetCylinderCreator()->GetCylinderSPhi() : 0. ;
}
G4double GateCylinderComponent::GetCylinderDPhi() const
{
  return GetCylinderCreator() ? GetCylinderCreator()->GetCylinderDPhi() : 0. ;
}

// Tool method: try to cast a creator into a box creator
GateCylinder* GateCylinderComponent::CastToCylinderCreator(GateVVolume* creator)
{ 
  return dynamic_cast<GateCylinder*>(creator); 
}


