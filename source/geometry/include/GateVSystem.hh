/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateVSystem_h
#define GateVSystem_h 1

#include "globals.hh"
#include <vector>

#include "GateClockDependent.hh"
#include "GateOutputVolumeID.hh"
#include "GateArrayComponent.hh"
#include "GateBoxComponent.hh"
#include "GateCylinderComponent.hh"
#include "GateWedgeComponent.hh"
#include "GatePulse.hh"
#include "GateDigi.hh"

class GateVolumeID;
class GateVVolume;
class GateDigi;

/*! \class  GateVSystem
    \brief  A GateVSystem is an integrated interface to a GATE geometry.
    \brief  It can read and return information on the geometry according to a predefined set-up 
    
    - GateVSystem - by Daniel.Strul@iphe.unil.ch (2002)
    
    - A system provides a pre-defined model of a geometry, such as a scanner or a source.
      This model is built as a tree of system-components (GateSystemComponent), starting
      from the 'tree-base' (m_baseComponent). In this model, each component has a specific
      role (detector head, crystal matrix, collimator...)
    
    - System components are activated when they are connected to an inserter of the geometry.
      Once a component is thus connected, it can read the inserter properties, such as its
      dimensions, position, movement parameters, number of copies...
      
    - For example, a typical PET scanner would incorporate a component for the detector 
      blocks ('rsector', 'block', 'bucket'...). This component would then be connected
      to the one geometry inserter that models this detector block. Once this connection
      is done, one can read the scanner properties that are related to the blocks: number
      of rings, number of blocks per ring, internal diameter...

    - A system is also responsible for computing output volume IDs (GateOutputVolumeID), which
      are used for data analysis and image reconstruction. This task is actually delegated
      to the component tree.
    
    - To see a concrete application of this mechanism, check the class GateCylindricalPETSystem
      
    - Note: from July to Oct. 2002, a system was a vector of system-levels. It was redesigned
      as a tree of system-components in Oct 2002.

    \sa GateCylindricalPETSystem, GateSystemComponent, GateOutputVolumeID
*/      
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateVSystem : public GateClockDependent
{
  public:
    /*! \brief Constructor

	\param itsName:       	the name chosen for this system
	\param isWithGantry:	tells whether there is a gantry (PET) or not (SPECT)
    */    
    GateVSystem(const G4String& itsName,G4bool isWithGantry);
    //| Destructor
    virtual ~GateVSystem();

    public:
    //! \name Description/print-out methods
    //@{

    //! Set the outputID name for a depth of the tree
       void SetOutputIDName(char * anOutputIDName, size_t depth);
    // For System Classes

    /*! \brief Method overloading the base-class virtual method Describe().
      	\brief This methods prints-out a description of the system

	\param indent: the print-out indentation (cosmetic parameter)
    */    
    virtual void Describe(size_t indent=0); 
    
      	   
    /*! \brief Virtual method to print a description of the system to a stream.
      	\brief It is essentially meant to be used by the class GateToLMF, but it may also be used by Describe()

	\param aStream: the output stream
	\param doPrintNumbers: tells whether we print-out the volume numbers in addition to their dimensions
    */    
    //virtual void PrintToStream(std::ostream& aStream,G4bool doPrintNumbers) {}
    virtual void PrintToStream(std::ostream& ,G4bool ) {}
    //@}

    //! \name Getters and setters
    //@{

    //! Compute the depth of the component tree
    size_t GetTreeDepth() const;

    //@}
     
    //! \name Component access methods
    //@{

    //! Get a pointer to the base of the component-tree
    GateSystemComponent* GetBaseComponent() const
      { return m_BaseComponent; }

    //! Define the base of the component tres
    void SetBaseComponent(GateSystemComponent* aBaseComponent)
      { m_BaseComponent = aBaseComponent; }
      
    //! Finds a component from its name
    GateSystemComponent* FindComponent(const G4String& componentName,G4bool silent=false) const; 

    //! template to find components of a specific type
    template <class C>
    C* FindTypedComponent(const G4String& aComponent) const;
    
    //! Finds an array-component from its name
    GateArrayComponent* FindArrayComponent(const G4String& aComponent) const;

    //! Finds a boxcreator-component from its name
    GateBoxComponent* FindBoxCreatorComponent(const G4String& aComponent) const;
 
    //! Finds a creatorcreator-component from its name
    GateCylinderComponent* FindCylinderCreatorComponent(const G4String& aComponent) const;
   
    //! Finds a wedgecreator-component from its name
    GateWedgeComponent* FindWedgeCreatorComponent(const G4String& aComponent) const;
 
    //! Returns the main-component of the system.
    virtual GateSystemComponent* GetMainComponent() const
    {  return m_mainComponentDepth ? m_BaseComponent->GetChildComponent(0) :  m_BaseComponent ; }

    //! Extract the ID of the main-component from a pulse
    //OK GND 2022 Obsolete TODO remove
    virtual G4int GetMainComponentID(const GatePulse& pulse)
    {  return pulse.GetComponentID(m_mainComponentDepth) ; }

    //GND OK 2022
    virtual G4int GetMainComponentIDGND(const GateDigi& digi);

    //! Returns the number of coincident-sector of the system.
    virtual size_t GetCoincidentSectorNumber()
    {  return GetMainComponent()->GetAngularRepeatNumber(); }
    
    //! Returns the number of coincident-sector of the system (for the spherical system Ecat Accel)
    virtual size_t GetCoincidentSectorNumberSphere()
    {  return GetMainComponent()->GetSphereAzimuthalRepeatNumber(); }
    
    //! Returns the detector-component (crystal, pixel...) of the system.
    virtual GateArrayComponent* GetDetectorComponent()
    {  return dynamic_cast<GateArrayComponent*>(GetMainComponent()->GetChildComponent(0)) ; }

    //! Extract the ID of the detector-component from a pulse
    virtual G4int GetDetectorComponentID(const GatePulse& pulse)
    {  return pulse.GetComponentID(m_mainComponentDepth+1) ; }

    //! Check whether an inserter is connected to the system
    //! (directly or through one of its ancestors).
    //! Returns true if the inserter belongs (directly or inderectly) to the system
    G4bool CheckConnectionToCreator(GateVVolume* anCreator) const;

    //@}

    //! Checks if all levels are defined in the system (need by readout and spatial resolution)
    G4bool CheckIfAllLevelsAreDefined();

    // Checks if all the ancestors of the lowest defined level are also defined. 
    // For spatial reslution to work with some undefined low levels.
    G4bool CheckIfEnoughLevelsAreDefined();

    //! Generate the output-volumeID based on the information stored in the volumeID
    virtual GateOutputVolumeID ComputeOutputVolumeID(const GateVolumeID& aVolumeID);

    //! Compute a subsection of an output-volumeID for the subtree starting from a component
    G4int ComputeSubtreeID(GateSystemComponent* component, const GateVolumeID& volumeID,
    			  GateOutputVolumeID& outputVolumeID,
			  size_t depth);

    //! Compute a single bin of an output ID for a component 
    virtual G4int ComputeComponentID(GateSystemComponent* aComponent, const GateVolumeID& volumeID);

    //! Compute a single bin of an output ID for a coincident component 
    virtual G4int ComputeMainComponentID(GateSystemComponent* aComponent, const GateVolumeID& volumeID);

    //! Compute a ring-ID from a coincident component ID
    inline virtual G4int ComputeRingID (G4int componentID)
    {  return componentID / GetCoincidentSectorNumber() ; }

    //! Compute a sector-ID from coincident component ID (for Ecat, CylindricalPET, CPET)
    inline virtual G4int ComputeSectorID(G4int componentID)
    {  return componentID % GetCoincidentSectorNumber() ; }
     
    //! Compute a sector-ID from coincident component ID (for spherical Ecat Accel system)
    inline virtual G4int ComputeSectorIDSphere(G4int componentID)
    {  return componentID % GetCoincidentSectorNumberSphere() ; }
    
    //! Get the name of the system
    inline G4String GetName()
    {   return mName ; }
    
    //! Get the own name of a system, note thate this name may be any name.
    inline G4String GetOwnName() const { return m_itsOwnName; }
    
    //Get the number of a system, this number is the order insertion number of a system and the systemID.
    inline G4int GetItsNumber() const { return m_itsNumber; }
    
    size_t ComputeNofElementsAtLevel(size_t level) const;
    size_t ComputeNofSubCrystalsAtLevel(size_t level, std::vector<G4bool>& enableList) const;
    size_t ComputeIdFromVolID(const GateOutputVolumeID& volID,std::vector<G4bool>& enableList) const;
    //G4ThreeVector ComputeObjectCenter(const std::vector<G4int>& numList) const;
    G4ThreeVector ComputeObjectCenter(const GateVolumeID* volID) const;
    GateVolumeID* MakeVolumeID(const std::vector<G4int>& numList) const;
  public:
    typedef std::vector< GateSystemComponent* > compList_t;
    compList_t* MakeComponentListAtLevel(G4int level) const;
  protected:
    GateSystemComponent * m_BaseComponent;      	//!< The base component of the system
    size_t m_mainComponentDepth;		//!< depth of the main component (0 or 1)
    G4String m_itsOwnName;                      //! a name of a system, may be any name (multi-system approach)
    G4int m_itsNumber;                          //! the insertion order of a system, it is too the systemID ((multi-system approach)
    G4int m_sysNumber;
    G4int static m_insertionOrder;              //! a static member to carry the insertion number (multi-system approach)
};


#endif

