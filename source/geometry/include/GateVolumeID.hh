/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVolumeID_h
#define GateVolumeID_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4AffineTransform.hh"

#include "GateVVolume.hh"

class G4TouchableHistory;


/*! \class GateVolumeSelector
    \brief Allows to retrieve a physical volume and its GateVVolume
*/      
class GateVolumeSelector
{
  public:
    
    //! Constructs a GateVolumeSelector for a physical volume
    GateVolumeSelector(G4VPhysicalVolume* itsVolume);
    
    virtual ~GateVolumeSelector() {}

  public:
    //! \name getters and setters
    //@{

    G4int  GetDaughterID() const     	      { return m_daughterID;}   //!< Get the daughterID
    GateVVolume* GetCreator() const     { return m_creator;}    	//!< Get the creator ptr
    G4int GetCopyNo() const     	      { return m_copyNo;}    	//!< Get the volume copy-number
    G4VPhysicalVolume* GetVolume() const     	   
      { return  m_creator->GetPhysicalVolume(m_copyNo); }    	      	//!< Get the physical volume

    //@}
   
    //! Comparison operator
    inline G4bool operator==(const GateVolumeSelector& right)  const
    { return (m_creator==right.m_creator) && (m_copyNo==right.m_copyNo);}

    //! printing method
    friend std::ostream& operator<<(std::ostream&, const GateVolumeSelector& volumeLevelID);    

  protected:
    G4int m_daughterID;
    GateVVolume* m_creator;  //! object-creator that created the volume
    G4int m_copyNo;   	      	      //! copy-no of the volume
};




/*! \class  GateVolumeID
    \brief  The GateVolumeID stores all the navigation information needed for reaching and
    \brief  identifying a specific real-space volume (i.e. a touchable)
    
    - GateVolumeID - by Daniel.Strul@iphe.unil.ch (May 7 2002)
    
    - A GateVolumeID stores a navigation path to a specific real-space volume as a vector of 
      GateVolumeSelector. 
      
    - Once a GateVolumeID has been created for a touchable, it can provide the following services:
      - Get any volume of the path (i.e. the touchable itself or any of its ancestors)
      - Transfer a position from a volume's reference frame into another volume's reference frame
      - Return a "daughterID" for each level
*/      
class GateVolumeID : public std::vector<GateVolumeSelector>
{
  public:

    //! Compute the GateVolumeID for a touchable
    GateVolumeID(const G4TouchableHistory* touchable);

    //! Creates an empty volumeID
    GateVolumeID();

    //! Recreate a volumeID from data stored in a hit file
    GateVolumeID(G4int *daughterID, size_t arraySize);

    virtual inline ~GateVolumeID() {}

  public:
    inline G4bool IsValid() const     { return size()!=0; }   	    //!< Returns true for a valid (i.e. not empty) volumeID
    inline G4bool IsInvalid() const   { return !IsValid(); }  	    //!< Returns true for an invalid (i.e. empty) volumeID

    inline G4bool IsValidDepth(size_t depth)   const 
    { return (depth<size());}  	    //!< Check whether a depth is within the vector range


    //! Get daughter ID at a position in the vector
    inline G4int GetDaughterID(size_t depth) const
      { return (IsValidDepth(depth)) ? GetSelector(depth).GetDaughterID() : -1;}    

    //! Retrieves a physical volume within the path
    G4VPhysicalVolume* GetVolume(size_t depth) const  	      	    
      { return (IsValidDepth(depth)) ? GetSelector(depth).GetVolume() : 0;}    
    
    inline G4VPhysicalVolume* GetTopVolume() const 
      { return GetVolume(0);} 	      	      	      	      	    //!< Retrieves the world volume
    inline G4VPhysicalVolume* GetBottomVolume() const 
      { return GetVolume(size()-1);}      	      	      	    //!< Retrieves the bottom physical volume
 
    //! Retrieves a volume creator within the path
    GateVVolume* GetCreator(size_t depth) const  	      	    
      { return (IsValidDepth(depth)) ? GetSelector(depth).GetCreator() : 0;}    

    inline GateVVolume* GetBottomCreator() const 
      { return GetCreator(size()-1);}      	      	      	    //!< Retrieves the bottom creator

    G4int GetCreatorDepth (G4String name) const;                   //!< Retrieves the depth of requested creator
    
     //! Retrieves a copy no within the path
    G4int GetCopyNo(size_t depth) const  	      	    
      { return (IsValidDepth(depth)) ? GetSelector(depth).GetCopyNo() : -1;}    
    
   //! Get volume selector at a position in the vector
    inline const GateVolumeSelector& GetSelector(size_t depth) const
      { return (*this)[depth]; }    

    //! Appends a new level at the end of the vector
    inline void InsertVolumeLevel(G4VPhysicalVolume* volume)
    { insert(begin(),GateVolumeSelector(volume)); }

 
    //! Store the daughterIDs into an array
    void StoreDaughterIDs(G4int* dest,size_t destSize) const
    {
      	  for (size_t i=0;i<destSize;i++)
      	    dest[i] = GetDaughterID(i);
    }

     //! Retrieves the affine transformation connecting connecting the bottom volume to one of its ancestors
    G4AffineTransform ComputeAffineTransform(G4int ancestorDepth) const;

    
    /*! \brief Move a position from the reference frame of the bottom-volume into the reference frame of one of its ancestors

	\param position:      	  the position to be transfered into the ancestor's reference frame
	\param ancestorLevel:     The level of an ancestor of the bottom-volume, located somewhere between the top and the bottom.
	      	      	      	  If no level is specified, the world-level is selected
    */    
    G4ThreeVector MoveToAncestorVolumeFrame(G4ThreeVector position,G4int ancestorLevel=0) const;



    /*! \brief Move a position from the reference frame of an ancestor into the reference frame of the bottom-volume

	\param position:      	  the local position to be transfered into the descendant's reference frame
	\param ancestorLevel:     The level of an ancestor of the bottom-volume, located somewhere between the top and the bottom.
	      	      	      	  If no level is specified, the world-level is selected
    */    
    G4ThreeVector MoveToBottomVolumeFrame(G4ThreeVector position,G4int ancestorLevel=0) const;
    
    
    /*! \brief Move a position from one reference frame into another 

	\param position:  a position to be transfered into another reference frame
	\param transform: the transformation linking the two reference frames
	\param flagDirect: if true, apply the transform. If false, apply its inverse.
    */    
    G4ThreeVector  MoveToFrameByTransform(G4ThreeVector position,const G4AffineTransform& transform,G4bool flagDirect) const;



    //! Transfers a local position, given in the reference frame of a volume, into the reference frame of its mother volume 
    G4ThreeVector MoveToMotherFrame(G4ThreeVector position,G4VPhysicalVolume* physicalVolume) const
    {  return MoveToFrameByTransform(position,GetVolumeAffineTransform(physicalVolume),true); }

 
    //! Tool function: returns the affine transform linking a volume's reference frame to its mother's reference frame
    static G4AffineTransform GetVolumeAffineTransform(G4VPhysicalVolume* physicalVolume);

    //! Printing methods
    friend std::ostream& operator<<(std::ostream&, const GateVolumeID& volumeID);    
};

inline GateVolumeID::GateVolumeID()
 : std::vector<GateVolumeSelector>()
{}


#endif

