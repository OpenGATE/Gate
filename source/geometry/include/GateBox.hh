/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBox_h
#define GateBox_h 1

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "GateObjectChildList.hh"

class G4Box;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;

#include "globals.hh"

class GateBoxMessenger;

class GateBox : public GateVVolume
{

public : 
   GateBox(const G4String& itsName,
		 G4bool acceptsChildren=true, 
		 G4int depth=0); 
  // Constructor
  GateBox(const G4String& itsName, const G4String& itsMaterialName,
      	      	G4double itsXLength, G4double itsYLength, G4double itsZLength,
		G4bool itsFlagAcceptChildren=true, G4int depth=0);

  //Destructor
   
  virtual ~GateBox();
   
  FCT_FOR_AUTO_CREATOR_VOLUME(GateBox)

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly); 

  virtual void DestroyOwnSolidAndLogicalVolume();
   
  //! Implementation of the pure virtual method GetHalfDimension() declared by the base-class
  //! Returns the half-size of the box along an axis (X=0, Y=1, Z=2)
  inline G4double GetHalfDimension(size_t axis)	  {return GetBoxHalfLength(axis); }

  //! Set the box length along X    
  void SetBoxXLength (G4double lengthXChoice);       
    
  //! Set the box length along Y
  void SetBoxYLength (G4double lengthYChoice);      
    
  //! Set the box length along Z
  void SetBoxZLength (G4double lengthZChoice);   
    
  //! Get the box length along an axis  (X=0, Y=1, Z=2)
  inline G4double GetBoxLength(size_t axis)      {return mBoxLength[axis];}
  //! Get the box length along X
  inline G4double GetBoxXLength()          	   {return GetBoxLength(0);}
  //! Get the box length along Y
  inline G4double GetBoxYLength()          	   {return GetBoxLength(1);}
  //! Get the box length along Z
  inline G4double GetBoxZLength()          	   {return GetBoxLength(2);}

  //! Get the half-box length along an axis  (X=0, Y=1, Z=2)
  inline G4double GetBoxHalfLength(size_t axis)   {return GetBoxLength(axis)/2.;}
  //! Get the box half-length along X
  inline G4double GetBoxXHalfLength()      	    {return GetBoxHalfLength(0);}
  //! Get the box half-length along Y
  inline G4double GetBoxYHalfLength()      	    {return GetBoxHalfLength(1);}
  //! Get the box half-length along Z
  inline G4double GetBoxZHalfLength()      	    {return GetBoxHalfLength(2);}


protected :
   
  G4Box* pBoxSolid;
  G4LogicalVolume* pBoxLog;
  G4VPhysicalVolume* pBoxPhys; 
  
   
private :
   
  G4double mBoxLength[3]; 	      	      	    //!< Box lengths along the 3 axes
   
  GateBoxMessenger* pMessenger; 
};
MAKE_AUTO_CREATOR_VOLUME(box,GateBox)

#endif
