/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateWedgeComponent_h
#define GateWedgeComponent_h 1

#include "GateSystemComponent.hh"

class GateWedge;

/*! \class  GateWedgeComponent
    \brief  A GateWedgeComponent is a subtype of GateSystemComponent
    \brief  It is meant to be connected to Wedge creators, such as crystals or detectors
    
    - GateWedgeComponent - by rannou@informatica.usach.cl (2007)
    
    - A GateWedgeComponent is derived from GateSystemComponent, and thus inherits all
      the properties of system-components. In addition, it includes some methods
      that are specific to Wedge-creator creators: CastToWedgeCreator(), GetWedgeCreator(),
      GetWedgeLength()
      
    - There is also a method IsValidAttachmentRequest(), that overloads the method from
      the GateSystemComponent base-class: this overloading allows to check that the
      creator is indeed connected to a Wedge creator
      
    \sa GateSystemComponent, GateWedge
*/      
class GateWedgeComponent  : public GateSystemComponent
{
  public:
   /*! \brief Constructor

	\param itsName:       	    the name chosen for this system-component
	\param itsMotherComponent:  the mother of the component (0 if top of a tree)
	\param itsSystem:           the system to which the component belongs
    */    
    GateWedgeComponent(const G4String& itsName,
      	      	      	    GateSystemComponent* itsMotherComponent,
		      	            GateVSystem* itsSystem);
    //! Destructor
    virtual ~GateWedgeComponent();
    
    //! Method overloading the method IsValidAttachmentRequest() of the base-class GateSystemComponent
    //! It tells whether an creator may be attached to this component
    //! In addition to the test performed by the base-class' method, it also
    //! checks that the creator is indeed connected to a Wedge creator
    virtual G4bool IsValidAttachmentRequest(GateVVolume* anCreator) const;
   
    //! Return the Wedge creator attached to our own creator
    inline GateWedge* GetWedgeCreator() const
      	{ return GetWedgeCreator(GetCreator()); }

    //! Return the length along an axis of the Wedge-creator attached to our creator
    G4double GetWedgeLength(size_t axis) const;

    //! Tool method: return the creator attached to an creator
    static inline GateWedge* GetWedgeCreator(GateVVolume* anCreator) 
      	{ return anCreator ? CastToWedgeCreator( anCreator->GetCreator() ) : 0 ; }
	
    //! Tool method: try to cast a creator into a Wedge creator
    static GateWedge* CastToWedgeCreator(GateVVolume* creator);
};

#endif

