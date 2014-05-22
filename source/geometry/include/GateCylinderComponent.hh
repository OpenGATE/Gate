/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCylinderComponent_h
#define GateCylinderComponent_h 1

#include "GateSystemComponent.hh"

class GateCylinder;

/*! \class  GateCylinderComponent
    \brief  A GateCylinderComponent is a subtype of GateSystemComponent
    \brief  It is meant to be connected to cylinder creators, such as gantries
    \brief  or curved crystal 
    
    
    - GateCylinderComponent - by Daniel.Strul@iphe.unil.ch
    
    - A GateCylinderComponent is derived from GateSystemComponent, and thus inherits all
      the properties of system-components. In addition, it includes some methods
      that are specific to cylinder creators
      
    - There is also a method IsValidAttachmentRequest(), that overloads the method from
      the GateSystemComponent base-class: this overloading allows to check that the
      creator is indeed connected to a cylinder creator
      
    \sa GateSystemComponent, GateCylinder
*/      
class GateCylinderComponent  : public GateSystemComponent
{
  public:
   /*! \brief Constructor

	\param itsName:       	    the name chosen for this system-component
	\param itsMotherComponent:  the mother of the component (0 if top of a tree)
	\param itsSystem:           the system to which the component belongs
    */    
    GateCylinderComponent(const G4String& itsName,
      	      	      	    GateSystemComponent* itsMotherComponent,
			  GateVSystem* itsSystem);
			  //:GateSystemComponent(itsName,itsMotherComponent,itsSystem){}
    //! Destructor
    virtual ~GateCylinderComponent();
    
    //! Method overloading the method IsValidAttachmentRequest() of the base-class GateSystemComponent
    //! It tells whether an creator may be attached to this component
    //! In addition to the test performed by the base-class' method, it also
    //! checks that the creator is indeed connected to a cylinder creator
  virtual G4bool IsValidAttachmentRequest(GateVVolume* anCreator) const;
   
    //! Return the cylinder creator attached to our own creator
    inline GateCylinder* GetCylinderCreator() const
      	{ return GetCylinderCreator(GetCreator()); }

    G4double GetCylinderHeight() const ;
    G4double GetCylinderRmin() const ;
    G4double GetCylinderRmax() const ;
    G4double GetCylinderSPhi() const ;
    G4double GetCylinderDPhi() const ;

    //! Tool method: return the creator attached to an creator
    static inline GateCylinder* GetCylinderCreator(GateVVolume* anCreator) 
      	{ return anCreator ? CastToCylinderCreator( anCreator->GetCreator() ) : 0 ; }
	
    //! Tool method: try to cast a creator into a cylinder creator
  static GateCylinder* CastToCylinderCreator(GateVVolume* creator);
};

#endif


