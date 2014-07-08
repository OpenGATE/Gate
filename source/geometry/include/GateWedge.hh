/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateWedge_h
#define GateWedge_h 1

#include "globals.hh"

#include "GateVVolume.hh"
#include "G4RunManager.hh"
#include "G4VVisManager.hh"
#include "G4Trap.hh"
#include "GateVolumeManager.hh"
#include "GateObjectChildList.hh"

class GateWedgeMessenger;
class G4Material;
class G4LogicalVolume;
class G4VPhysicalVolume;

class GateWedge  : public GateVVolume
{
  public:
    //! Constructor    
    
    GateWedge(const G4String& itsName,
		    G4bool acceptsChildren=true, 
		    G4int depth=0); 
		 
    GateWedge(const G4String& itsName,const G4String& itsMaterialName,
      	      	    G4double itsXLength, G4double itsNarrowerXLength, G4double itsYLength,G4double itsZLength);
    //! Destructor
    virtual ~GateWedge();
    
    FCT_FOR_AUTO_CREATOR_VOLUME(GateWedge)

  public:

     virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly);
     virtual void DestroyOwnSolidAndLogicalVolume();  

     virtual void DescribeMyself(size_t indent);

     inline G4double GetHalfDimension(size_t axis)	  {return GetWedgeHalfLength(axis); }

     inline G4double ComputeMyOwnVolume()  const	 
       {return m_WedgeLength[0]*m_WedgeLength[1]*m_WedgeLength[2]; }


     //! Set the Wedge length along X
     void SetWedgeXLength (G4double val)   
     {  m_WedgeLength[0] = val; /*ComputeParameters();*/ }
     //! Set the Wedge length along narrower X
     void SetWedgeNarrowerXLength (G4double val)   
     {  m_WedgeLength[1] = val; /*ComputeParameters();*/ }
     //! Set the Wedge length along Y
     void SetWedgeYLength (G4double val)   
     {  m_WedgeLength[2] = val; /*ComputeParameters();*/ }
     //! Set the Wedge length along Z
     void SetWedgeZLength (G4double val)   
     {  m_WedgeLength[3] = val; /*ComputeParameters();*/ }
     

     //! Get the Wedge length along an axis  (X=0, Narrower X=1, Y=2, Z=3)
     inline G4double GetWedgeLength(size_t axis)       {return m_WedgeLength[axis];}
     //! Get the Wedge length along X
     inline G4double GetWedgeXLength()          	   {return GetWedgeLength(0);}
     //! Get the Wedge length along narrower X
     inline G4double GetWedgeNarrowerXLength()         {return GetWedgeLength(1);}
     //! Get the Wedge length along Y
     inline G4double GetWedgeYLength()          	   {return GetWedgeLength(2);}
     //! Get the Wedge length along Z
     inline G4double GetWedgeZLength()          	   {return GetWedgeLength(3);}

     //! Get the half-Wedge length along an axis  (X=0, Narrower X=1, Y=2, Z=3)
     inline G4double GetWedgeHalfLength(size_t axis)      {return GetWedgeLength(axis)/2.;}
     //! Get the Wedge half-length along X
     inline G4double GetWedgeXHalfLength()      	      {return GetWedgeHalfLength(0);}
     //! Get the Wedge half-length along narrower X
     inline G4double GetWedgeNarrowerXHalfLength()        {return GetWedgeHalfLength(1);}
     //! Get the Wedge half-length along Y
     inline G4double GetWedgeYHalfLength()      	      {return GetWedgeHalfLength(2);}
     //! Get the Wedge half-length along Z
     inline G4double GetWedgeZHalfLength()      	      {return GetWedgeHalfLength(3);}
 
    //@}

  private:
    //! \name own geometry
    //@{
     G4Trap*               m_Wedge_solid;       //!< Solid pointer
     G4LogicalVolume*      m_Wedge_log; 	    //!< logical volume pointer
    //@}

    //! \name parameters
    //@{
     G4double m_WedgeLength[4]; 	      	    //!< Wedge lengths along the 3 axes plus narrower side
    //@}

     //! Messenger
     GateWedgeMessenger* m_Messenger; 

};

MAKE_AUTO_CREATOR_VOLUME(wedge,GateWedge)

#endif

