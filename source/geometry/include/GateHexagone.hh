/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateHexagone_h
#define GateHexagone_h 1

#include "globals.hh"

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"

class GatePolyhedra;
class GateHexagoneMessenger;

class GateHexagone  : public GateVVolume
{
  public:
  
     GateHexagone(const G4String& itsName,
		  G4bool acceptsChildren=true, 
		  G4int depth=0); 
 
     GateHexagone(const G4String& itsName,const G4String& itsMaterialName,
      	      	      	G4double itsRadius,G4double itsHeight);
     virtual ~GateHexagone();

     FCT_FOR_AUTO_CREATOR_VOLUME(GateHexagone)

  public:
     virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool flagUpdateOnly);     
     virtual void DestroyOwnSolidAndLogicalVolume();
     virtual void DescribeMyself(size_t indent);
     
     inline G4double GetHalfDimension(size_t axis) 
     { if (axis==2)
      	  return m_HexagoneHeight/2.;
       else
      	  return m_HexagoneRadius;
     }

  public:
     inline G4double GetHexagoneHeight()          	      {return m_HexagoneHeight;}
     inline G4double GetHexagoneRadius()          	      {return m_HexagoneRadius;}
     
     inline void SetHexagoneRadius(G4double val)      	      { m_HexagoneRadius = val; /*ComputeParameters();*/}
     inline void SetHexagoneHeight(G4double val)      	      { m_HexagoneHeight = val; /*ComputeParameters();*/ }

  private:
     G4double m_HexagoneRadius;
     G4double m_HexagoneHeight;
     
     G4double* m_HexagoneZPlane;
     G4double* m_HexagoneRInner;
     G4double* m_HexagoneROuter;

     //! own geometry
     GatePolyhedra*       m_Hexagone_solid;
     G4LogicalVolume*     m_Hexagone_log;

     //! parameters
     //! Messenger
     GateHexagoneMessenger* m_Messenger; 

};
MAKE_AUTO_CREATOR_VOLUME(hexagone,GateHexagone)

#endif


