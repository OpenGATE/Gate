/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBoxComponent_h
#define GateBoxComponent_h 1

#include "GateSystemComponent.hh"

class GateBox;

/*! \class  GateBoxComponent
    \brief  A GateBoxComponent is a subtype of GateSystemComponent
    \brief  It is meant to be connected to box Creators, such as crystals or detectors
    
    - GateBoxComponent - by Daniel.Strul@iphe.unil.ch (Oct 2002)
    
    - A GateBoxComponent is derived from GateSystemComponent, and thus inherits all
      the properties of system-components. In addition, it includes some methods
      that are specific to box-creator Creators: CastToBoxCreator(), GetBoxCreator(),
      GetBoxLength()
      
    - There is also a method IsValidAttachmentRequest(), that overloads the method from
      the GateSystemComponent base-class: this overloading allows to check that the
      Creator is indeed connected to a box creator
      
    \sa GateSystemComponent, GateBox
*/      
class GateBoxComponent  : public GateSystemComponent
{
  public:
   /*! \brief Constructor

	\param itsName:       	    the name chosen for this system-component
	\param itsMotherComponent:  the mother of the component (0 if top of a tree)
	\param itsSystem:           the system to which the component belongs
    */    
    GateBoxComponent(const G4String& itsName,
      	      	      	    GateSystemComponent* itsMotherComponent,
		      	    GateVSystem* itsSystem);
    //! Destructor
    virtual ~GateBoxComponent();
    
    //! Method overloading the method IsValidAttachmentRequest() of the base-class GateSystemComponent
    //! It tells whether an Creator may be attached to this component
    //! In addition to the test performed by the base-class' method, it also
    //! checks that the Creator is indeed connected to a box creator
    virtual G4bool IsValidAttachmentRequest(GateVVolume* anCreator) const;
   
    //! Return the box creator attached to our own Creator
    inline GateBox* GetBoxCreator() const
      	{  return GetBoxCreator(GetCreator()); }

    //! Return the length along an axis of the box-creator attached to our Creator
    G4double GetBoxLength(size_t axis) const;

    //! Tool method: return the creator attached to an Creator
    static inline GateBox* GetBoxCreator(GateVVolume* anCreator) 
      	{  return anCreator ? CastToBoxCreator( anCreator->GetCreator() ) : 0 ; }
	
    //! Tool method: try to cast a creator into a box creator
    static GateBox* CastToBoxCreator(GateVVolume* creator);
};

#endif

