/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCylinder.hh"
#include "GateCylinderMessenger.hh"

#include "G4Tubs.hh"
#include "G4UnitsTable.hh"
#include "G4Colour.hh"

#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4ThreeVector.hh"



#include "globals.hh"

class G4Material;

  
//---------------------------------------------------------------------------------
// Initialisation
// Constructor
//---------------------------------------------------------------------------------
GateCylinder::GateCylinder(const G4String& itsName, const G4String& /*itsMaterialName*/,//DS comment to avoid warning
      	      	              	            G4double itsRmax, G4double itsHeight,
				            G4double itsRmin,
	              	      	            G4double itsSPhi, G4double itsDPhi,
				            G4bool itsFlagAcceptChildren, G4int depth)				      
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    pCylinderSolid(0), pCylinderLog(0), 
    mCylinderHeight(itsHeight),
    mCylinderRmin(itsRmin), mCylinderRmax(itsRmax),
    mCylinderSPhi(itsSPhi), mCylinderDPhi(itsDPhi) 
   {
   
    pMessenger = new GateCylinderMessenger(this);
    
   }
//--------------------------------------------------------------------------------- 


//---------------------------------------------------------------------------------
   GateCylinder::GateCylinder(const G4String& itsName,
		                            G4bool itsFlagAcceptChildren,
			                    G4int depth)				      
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    pCylinderSolid(0), pCylinderLog(0)
   {
    mCylinderHeight = 1.0*cm; 
    mCylinderRmin = 0.0*cm;
    mCylinderRmax = 1.0*cm;
    mCylinderSPhi = 0.;
    mCylinderDPhi = 2*M_PI;
    pMessenger = new GateCylinderMessenger(this);
   }
//--------------------------------------------------------------------------------- 


//---------------------------------------------------------------------------------
//Destuctor
//---------------------------------------------------------------------------------
   GateCylinder::~GateCylinder()
   {
     
     delete pMessenger;
     
   }
//--------------------------------------------------------------------------------- 


//---------------------------------------------------------------------------------
G4LogicalVolume* GateCylinder::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  
  if (!flagUpdateOnly || !pCylinderSolid) {
    // Build mode: build the solid, then the logical volume
    pCylinderSolid
      = new G4Tubs(GetSolidName(), 
      		   mCylinderRmin, 
	  	   mCylinderRmax,
		   mCylinderHeight/2.,
		   mCylinderSPhi,
		   mCylinderDPhi);
    pCylinderLog
      = new G4LogicalVolume(pCylinderSolid, mater, GetLogicalVolumeName(),0,0,0);

  } 
  else {
    // Update mode: refresh the dimensions of the solid
	    pCylinderSolid->SetZHalfLength(GetCylinderHalfHeight());
	    pCylinderSolid->SetInnerRadius(GetCylinderRmin());
	    pCylinderSolid->SetOuterRadius(GetCylinderRmax());
	    pCylinderSolid->SetStartPhiAngle(GetCylinderSPhi());
	    pCylinderSolid->SetDeltaPhiAngle(GetCylinderDPhi());
  }
  
  return pCylinderLog;
 
}
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
void GateCylinder::DestroyOwnSolidAndLogicalVolume()
{
    
  if (pCylinderLog)
    delete pCylinderLog;
    pCylinderLog = 0;

  if (pCylinderSolid)
    delete pCylinderSolid;
    pCylinderSolid = 0;

}
//---------------------------------------------------------------------------------
