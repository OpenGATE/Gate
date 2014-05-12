/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCylinder_h
#define GateCylinder_h 1

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

#include "globals.hh"

class G4Tubs;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4PVPlacement;
class G4Material;

class GateCylinderMessenger;


class GateCylinder : public GateVVolume
{
  
   public : 
   
   GateCylinder(const G4String& itsName,
		       G4bool acceptsChildren=true, 
		       G4int depth=0);
   // Constructor
    GateCylinder(const G4String& itsName,const G4String& itsMaterialName,
      	      	         G4double itsRmax, G4double itsHeight,
		         G4double itsRmin=0.,
	                 G4double itsSPhi=0., G4double itsDPhi=2*M_PI, 
			 G4bool acceptsChildren=true, 
		         G4int depth=0);
  
    //Destructor 
    virtual ~GateCylinder();
    
    FCT_FOR_AUTO_CREATOR_VOLUME(GateCylinder)
    
    virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
    
    
    //! Implementation of the pure virtual method DestroyOwnSolidAndLogicalVolume() declared by the base-class.
    //! Destroy the solid, logical and physical volumes created by ConstructOwnVolume()
    virtual void DestroyOwnSolidAndLogicalVolume();  
   
    //! \brief Implementation of the virtual method DescribeMyself(), to print-out
    //  \brief a description of the creator 
//    virtual void DescribeMyself(size_t indent); 
  
    //! Implementation of the pure virtual method GetHalfDimension() declared by the base-class
     //! Must return the half-size of the cylinder along an axis (X=0, Y=1, Z=2)
     //! Returns the radiusor the height depending on the axis: accurate only for full cylinders
     inline G4double GetHalfDimension(size_t axis) 
     { if (axis==2)
      	  return GetCylinderHalfHeight();
       else
      	  return mCylinderRmax;
     }

     //! Overload of the dummy virtual method ComputeMyOwnVolume() defined by the base-class
     //! Returns the volume of the solid
//     G4double ComputeMyOwnVolume()  const;	 


    //! \name getters and setters
    //@{

     //! Get the height
     inline G4double GetCylinderHeight()      {return mCylinderHeight;};
     //! Get the half of the height
     inline G4double GetCylinderHalfHeight()  {return mCylinderHeight/2.;};
     //! Get the internal diameter
     inline G4double GetCylinderRmin()        {return mCylinderRmin;};
     //! Get the external diameter
     inline G4double GetCylinderRmax()        {return mCylinderRmax;};
     //! Get the start phi angle
     inline G4double GetCylinderSPhi()        {return mCylinderSPhi;};
     //! Get the angular span for the phi angle
     inline G4double GetCylinderDPhi()        {return mCylinderDPhi;};

     //! Set the height
     void SetCylinderHeight   (G4double val) 
      	//{ pcylinderHeight = val; ComputeParameters(); }
	{ mCylinderHeight = val;}
     //! Set the internal diameter
     void SetCylinderRmin  (G4double val) 
      	//{ pcylinderRmin = val; ComputeParameters(); }
	{ mCylinderRmin = val;}
     //! Set the external diameter
     void SetCylinderRmax  (G4double val) 
      	//{  pcylinderRmax = val; ComputeParameters(); }
	{  mCylinderRmax = val;}
     //! Set the start phi angle
     void SetCylinderSPhi  (G4double val) 
      	//{  pcylinderSPhi = val; ComputeParameters(); }
	 {  mCylinderSPhi = val;}
     //! Set the angular span for the phi angle
     void SetCylinderDPhi  (G4double val) 
      	//{  pcylinderDPhi = val; ComputeParameters(); }
	 {  mCylinderDPhi = val;}

    //@}
    
   private :
   
   G4Tubs* pCylinderSolid;
   G4LogicalVolume* pCylinderLog;
//   G4VPhysicalVolume* pCylinderPhys; 
   
   //@{
   G4double mCylinderHeight;   	      	    //!< height
   G4double mCylinderRmin;   	      	      	    //!< internal diameter
   G4double mCylinderRmax;   	      	      	    //!< external diameter
   G4double mCylinderSPhi;   	      	      	    //!< start phi angle
   G4double mCylinderDPhi;   	      	      	    //!< angular span for the phi angle
   //@}
  
   //! Messenger
   GateCylinderMessenger* pMessenger;  
};

MAKE_AUTO_CREATOR_VOLUME(cylinder,GateCylinder)

#endif
