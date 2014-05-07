/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateElTub_h
#define GateElTub_h 1

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

#include "globals.hh"


class G4EllipticalTube;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4PVPlacement;
class G4Material;

class GateElTubMessenger;


class GateElTub : public GateVVolume
{
  
   public : 
   
   GateElTub(const G4String& itsName,
		       G4bool acceptsChildren=true, 
		       G4int depth=0);
   // Constructor
    GateElTub(const G4String& itsName,const G4String& itsMaterialName,
      	      	         G4double itsRlong, G4double itsHeight,
		         G4double itsRshort=1.,
			 G4bool acceptsChildren=true, 
		         G4int depth=0);
  
    //Destructor 
    virtual ~GateElTub();
    
    FCT_FOR_AUTO_CREATOR_VOLUME(GateElTub)
    
    virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
    
    
    //! Implementation of the pure virtual method DestroyOwnSolidAndLogicalVolume() declared by the base-class.
    //! Destroy the solid, logical and physical volumes created by ConstructOwnVolume()
    virtual void DestroyOwnSolidAndLogicalVolume();  
   
    //! \brief Implementation of the virtual method DescribeMyself(), to print-out
    //  \brief a description of the creator 
//    virtual void DescribeMyself(size_t indent); 
  
    //! Implementation of the pure virtual method GetHalfDimension() declared by the base-class
     //! Must return the half-size of the ElTub along an axis (X=0, Y=1, Z=2)
     //! Returns the radiusor the height depending on the axis: accurate only for full ElTubs
     inline G4double GetHalfDimension(size_t axis) 
     { if (axis==2)
      	  return GetElTubHalfHeight();
       else
      	  return mElTubRlong;
     }

     //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
     //! Returns the volume of the solid
//     G4double ComputeMyOwnVolume()  const;	 


    //! \name getters and setters
    //@{

     //! Get the height
     inline G4double GetElTubHeight()      {return mElTubHeight;};
     //! Get the half of the height
     inline G4double GetElTubHalfHeight()  {return mElTubHeight/2.;};
     //! Get the internal diameter
     inline G4double GetElTubRshort()        {return mElTubRshort;};
     //! Get the external diameter
     inline G4double GetElTubRlong()        {return mElTubRlong;};

     //! Set the height
     void SetElTubHeight   (G4double val) 
      	//{ pElTubHeight = val; ComputeParameters(); }
	{ mElTubHeight = val;}
     //! Set the internal diameter
     void SetElTubRshort  (G4double val) 
      	//{ pElTubRshort = val; ComputeParameters(); }
	{ mElTubRshort = val;}
     //! Set the external diameter
     void SetElTubRlong  (G4double val) 
      	//{  pElTubRlong = val; ComputeParameters(); }
	{  mElTubRlong = val;}

    //@}
    
   private :
   
   G4EllipticalTube* pElTubSolid;
   G4LogicalVolume* pElTubLog;
//   G4VPhysicalVolume* pElTubPhys; 
   
   //@{
   G4double mElTubHeight;   	      	    //!< height
   G4double mElTubRshort;   	      	      	    //!< internal diameter
   G4double mElTubRlong;   	      	      	    //!< external diameter
   //@}
  
   //! Messenger
   GateElTubMessenger* pMessenger;  
};

MAKE_AUTO_CREATOR_VOLUME(eltub,GateElTub)

#endif
