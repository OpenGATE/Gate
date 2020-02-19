/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateSystemComponent_h
#define GateSystemComponent_h 1

#include "globals.hh"
#include <vector>

#include "G4ThreeVector.hh"

#include "GateClockDependent.hh"
#include "GateVVolume.hh"
#include "GateSystemComponentList.hh"
#include "GateObjectRepeaterList.hh"
class GateVSystem;
class GateSystemComponentMessenger;
class GateTranslationMove;
class GateRotationMove;
class G4VPhysicalVolume;
class GateVolumePlacement;
class GateLinearRepeater;
class GateAngularRepeater;
class GateGenericRepeater;
class GateSphereRepeater;
class GateOrbitingMove;
class GateEccentRotMove;


/*! \class  GateSystemComponent
    \brief  A GateSystemComponent is a level of a system (GateVSystem)
    
    - GateSystemComponent - by Daniel.Strul@iphe.unil.ch (Oct 2002)
    
    - A system component is part of of a component tree: it handles a tree of subcomponents,
      and may be a daughter of some higher-level component. The whole tree defines a 'system',
      i.e. a geometry model: in this model, each component has a predefined role (detector
      head, crystal matrix, collimator...)
    
    - A component tree provides a pre-defined model of a geometry, such as a scanner or a 
      detector block. In this model, each component has a specific role (detector head, 
      crystal matrix, collimator...)
    
    - System components are activated when they are connected to an creator of the geometry.
      Once a component is thus connected, it can read the creator properties, such as its
      dimensions, position, movement parameters, number of copies...
      
    - For example, a typical PET scanner would incorporate a component for the detector 
      blocks ('rsector', 'block', 'bucket'...). This component would then be connected
      to the one geometry creator that models this detector block. Once this connection
      is done, one can read the scanner properties that are related to the blocks: number
      of rings, number of blocks per ring, internal diameter... Along with all its subcomponents,
      thus block-component would allow to read all the detector properties.

    - Note that the component methods are of two distinct types: for some of them, the component
      just answers for itself; for others, it recursively calls the methods of all its subcomponents

    - Note: from July to Oct. 2002, a system was a vector of system-levels, each level managing a
      vector of components. On Oct. 2002, the whole system (which was too complex) was redesigned, 
      and the current mechanism (tree of system-components) replaced the previous one (vector of
      system-levels)

    \sa GateVSystem
*/      
class GateSystemComponent  : public GateClockDependent
{
  public:
    /*! \brief Constructor

	\param itsName:       	    the name chosen for this system-component
	\param itsMotherComponent:  the mother of the component (0 if top of a tree)
	\param itsSystem:           the system to which the component belongs
    */    
    GateSystemComponent(const G4String& itsName,
      	      	      	GateSystemComponent* itsMotherComponent,
		      	GateVSystem* itsSystem);
    //! Destructor
    virtual ~GateSystemComponent();
    
 
    //! \name Description/print-out methods
    //@{

    /*! \brief Method overloading the base-class virtual method Describe().
      	\brief This methods prints-out a description of the system component

	\param indent:        	    the print-out indentation (cosmetic parameter)
    */    
    virtual void Describe(size_t indent=0);
    //@}

    //! \name Getters and setters
    //@{

    //! Get the system to which the component belongs
    inline GateVSystem* GetSystem() const
      { return m_system; }

    //! Get the mother of the component
    GateSystemComponent* GetMotherComponent() const 
      {  return m_motherComponent;}

    //! Set the mother of the component
    void SetMotherComponent(GateSystemComponent* aComponent)
      {  m_motherComponent = aComponent; }

    //! Check whether we're active, i.e. connected to an creator
    G4bool IsActive() const
      { return (m_creator!=0);}

    //! Compute the total number of volumes handled by the 'younger' sister-components
    virtual G4int ComputeOutputOffset();

    //@}

		void SetminSectorDiff( G4int aI ){ m_minSectorDiff = aI; };
    G4int GetminSectorDiff(){ return m_minSectorDiff; };
    void setInCoincidenceWith(G4String );    G4int IsInCoincidenceWith(G4String );
    void SetRingID(G4int aR ){ m_ringID = aR;};    G4int GetRingID(){ return m_ringID;};


    //! \name Access to creator properties
    //@{

    //! Returns the creator attached to the component
    inline virtual GateVVolume* GetCreator() const   	
      { return m_creator; }


    //! Attach an creator to the component, provided that the attachment request is valid
    //! (calls IsValidAttachmentRequest() to check the request validity)
    virtual void SetCreator(GateVVolume* anCreator);

    /*! \brief Check whether an creator is connected to the component tree
      	
	\param anCreator: the creator we want to check
	
	\return true if the creator is attached to one of the components of the component-tree
    */
    G4bool CheckConnectionToCreator(GateVVolume* anCreator);
    
    
    //! Tells whether an creator may be attached to this component
    //! This virtual method makes a number of tests: is the creator pointer valid,
    //! does the creator owns a movement-list, does it own a repeater-list...
    //! It can be and should be overloaded when we want to do specific tests (for specific components)
    virtual G4bool IsValidAttachmentRequest(GateVVolume*
    anCreator) const;
   
    //! Returns the number of physical volumes created by the creator
    virtual size_t GetVolumeNumber() const;

    //! Tenplate for finding moves of a specific type: the function returns a non-zero pointer
    //! only if a move of the requested type was found in the creator's move list
    template <class C>
    C* FindMove() const;

    //! Tenplate for finding repeaters of a specific type: the function returns a non-zero pointer
    //! only if a repeater of the requested type was found in the creator's repeater list
    template <class C>
    C* FindRepeater() const;

    //@}


    //! \name Access to physical volumes' properties
    //@{

     //! Returns one of the physical volumes created by the creator
     virtual G4VPhysicalVolume* GetPhysicalVolume(size_t copyNumber=0) const ;
     //! Returns the translation vector for one of the physical volumes created by the creator
     virtual G4ThreeVector  GetCurrentTranslation(size_t copyNumber=0) const;
     //! Returns the rotation matrix for one of the physical volumes created by the creator
     virtual G4RotationMatrix* 	   GetCurrentRotation(size_t copyNumber=0) const;
    
    //@}


    //! \name Access to the placement move properties (if any)
    //@{

    //! The function returns the creator's placement move, if one can be find was found in the creator's move list
    GateVolumePlacement* FindPlacementMove() const;

    //@}


    //! \name Access to the translation move properties (if any)
    //@{

    //! The function returns the creator's translation move, if one can be find was found in the creator's move list
    GateTranslationMove* FindTranslationMove() const ;
    //! The function returns the creator's translation velocity, if a translation can be find was found in the creator's move list
    G4ThreeVector GetTranslationVelocity() const;

    //@}


    //! \name Access to the rotation move properties (if any)
    //@{

    //! The function returns the creator's rotation move, if one can be find was found in the creator's move list
    GateRotationMove* FindRotationMove() const;
    //! The function returns the creator's rotation velocity, if a rotation can be find was found in the creator's move list
    G4double GetRotationVelocity() const;
    
    //@}


    //! \name Access to the orbiting move properties (if any)
    //@{

    //! The function returns the creator's orbiting move, if one can be find was found in the creator's move list
    GateOrbitingMove* FindOrbitingMove() const;
    //! The function returns the creator's orbiting velocity, if an orbiting can be find was found in the creator's move list
    G4double GetOrbitingVelocity() const;
 
    //! \name Access to the EccentRot move properties (if any)
    //@{

    //! The function returns the creator's EccentRotMove move, if one can be find was found in the creator's move list
    GateEccentRotMove* FindEccentRotMove() const;

    //! The function returns the creator's EccentRot velocity, if an orbiting can be find was found in the creator's move list
    G4double GetEccentRotVelocity() const;

    //! The function returns the creator's shift position, if a EccentRot can be find was found in the creator's move list
    const G4ThreeVector& GetEccentRotShift() const;
    
    
    //@}


    //! \name Access to the linear repeater properties (if any)
    //@{

    //! Finds the first linear-repeater in the creator's repeater list
    GateLinearRepeater* FindLinearRepeater();
    //! Finds the first linear-repeater's repeat number
    G4int GetLinearRepeatNumber();
    //! Finds the first linear-repeater's repeat vector
    const G4ThreeVector& GetLinearRepeatVector();

    //@}


    //! \name Access to the angular repeater properties (if any)
    //@{

    //! Finds the first angular-repeater in the creator's repeater list
    GateAngularRepeater* FindAngularRepeater();
    //! Finds the first angular-repeater's repeat number
    G4int GetAngularRepeatNumber();
    //! Finds the first angular-repeater's repeat angular pitch
    G4double GetAngularRepeatPitch();

    //! Finds the first angular-repeater's modulo number
    G4int GetAngularModuloNumber();

    //! Finds the first angular-repeater's repeat Z1..Z8 shift
    G4double GetAngularRepeatZShift1();
    G4double GetAngularRepeatZShift2();
    G4double GetAngularRepeatZShift3();
    G4double GetAngularRepeatZShift4();
    G4double GetAngularRepeatZShift5();
    G4double GetAngularRepeatZShift6();
    G4double GetAngularRepeatZShift7();
    G4double GetAngularRepeatZShift8();


    //! \name Access to the sphere repeater properties (if any)
    //@{

    //! Finds the first sphere-repeater in the creator's repeater list
    GateSphereRepeater* FindSphereRepeater();
    //! Finds the first sphere-repeater's repeat axial pitch 
    G4double GetSphereAxialRepeatPitch();
    //! Finds the first sphere-repeater's repeat azimuthal pitch 
    G4double GetSphereAzimuthalRepeatPitch();
    //! Finds the first sphere-repeater's axial repeat number
    G4int GetSphereAxialRepeatNumber();
    //! Finds the first sphere-repeater's azimuthal repeat number
    G4int GetSphereAzimuthalRepeatNumber();
    //! Finds the first sphere-repeater's radius of replication
    G4double GetSphereRadius();


    //@}


    //@}

    //! \name Access to the generic repeater properties (if any)
    //@{

    //! Finds the first generic repeater in the creator's repeater list
    GateGenericRepeater* FindGenericRepeater();
    //! Finds the first generic repeater's repeat number
    G4int GetGenericRepeatNumber();

    //@}

    
    //! \name Computation of offsets to get radiuses
    //@{

    //! Alignment constants used by ComputeOffset()
    enum Alignment1D {
        align_right=-1,       
        align_center = 0,
      	align_left=+1,
	align_unknown=999
      };

    /*! \brief	Compute the offset (displacement) between a feature of the creator and a feature of its mother creator
    
      	By default, all alignments are set to align_center, so that we compute the offset between the creator's center 
	and the center of its mother's reference frame
	We could select align_left for both alignments: in that case, we would compute the offset between the left edge
	of the creator and the left edge of its mother
	To compute an internal ring diameter from a block position, we actually select align_left for the block and
	align_center for its mother: thus, we compute the distance between the block's left edge and its mother's center
    
      	\param	axis: 	      	      the axis along which we want to compute the offset
	\param	alignment:    	      the feature of the creator for which we want to compute the offset
	      	      	      	      it can be its center (align_center), its left border (align_left) or its right border (align_right)
      	\param	referenceAlignment:   the feature of the mother volume, with regards to which we compute the offset
	      	      	      	      it can be its center (align_center), its left border (align_left) or its right border (align_right)

      	\return the offset along the requested axis between the creator's feature considered and the mother's feature considered
    */
    G4double ComputeOffset(size_t axis,Alignment1D alignment=align_center,Alignment1D referenceAlignment=align_center) const;

    //@}


    //! \name Access to properties and objects of the component tree
    //@{

    //! Compute the total depth of the component tree
    //! This depth is at least 1 (the 'depth' of the component on its own)
    //! If the component has children, the depth is 1 + the distance between the component and its deepest descendant
    size_t GetTreeDepth() const
    {  return 1 + m_childComponentList->GetMaxChildTreeDepth(); }

    //! Return the number of daughters in the child list
    size_t GetChildNumber() const
      { return m_childComponentList->GetChildNumber(); }

    //! Return one of the daughter-components
    GateSystemComponent* GetChildComponent(size_t i) const 
      {  return m_childComponentList->GetChildComponent(i); }

    //! Append a new component at the end of the child-list
    void InsertChildComponent(GateSystemComponent* aComponent)
    { m_childComponentList->InsertChildComponent(aComponent); }

    //! Compute the number of active daughter-components (i.e. components that are linked to an creator)
    size_t GetActiveChildNumber() const
    {  return m_childComponentList->GetActiveChildNumber(); }
      
    //! Finds a component in the component tree from its name
    GateSystemComponent* FindSystemComponent(const G4String& componentName);

    //@}


  protected:
    GateVSystem       	      	    *m_system;    	   //!< System to which the component belongs
    GateVVolume               *m_creator;           //!< Creator attached to the component
    GateSystemComponentMessenger    *m_messenger; 	   //!< Messenger
    GateSystemComponent       	    *m_motherComponent;    //!< Mother of the component
    GateSystemComponentList         *m_childComponentList; //!< List of daughter components (next level of the component tree)

	G4int m_minSectorDiff;
    std::vector<G4String> m_coincidence_rsector; // for rsector only 
    G4int m_ringID;
};


// Tenplate for finding moves of a specific type: the function returns a non-zero pointer
// only if a move of the requested type was found in the creator's move list
template <class C>
C* GateSystemComponent::FindMove() const
{
      if (!m_creator)
      	return 0;
      GateObjectRepeaterList* aList = m_creator->GetMoveList();
      for (size_t i=0; i < aList->size() ; i++)
      	if ( dynamic_cast<C*>(aList->GetRepeater(i)) )
      	  return dynamic_cast<C*>(aList->GetRepeater(i));
      return 0;
}


// Tenplate for finding repeaters of a specific type: the function returns a non-zero pointer
// only if a repeater of the requested type was found in the creator's repeater list
template <class C>
C* GateSystemComponent::FindRepeater() const
{
      if (!m_creator)
      	return 0;
      GateObjectRepeaterList* aList = m_creator->GetRepeaterList();
      for (size_t i=0; i < aList->size() ; i++)
      	if ( dynamic_cast<C*>(aList->GetRepeater(i)) )
      	  return dynamic_cast<C*>(aList->GetRepeater(i));
      return 0;
}



#endif

