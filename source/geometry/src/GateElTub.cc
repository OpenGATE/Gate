/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateElTub.hh"
#include "GateElTubMessenger.hh"

#include "G4EllipticalTube.hh"

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
GateElTub::GateElTub(const G4String& itsName, const G4String& /*itsMaterialName*/,//DS comment to avoid warning
      	      	              	            G4double itsRlong, G4double itsHeight,
				            G4double itsRshort,
				            G4bool itsFlagAcceptChildren, G4int depth)				      
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    pElTubSolid(0), pElTubLog(0), 
    mElTubHeight(itsHeight),
    mElTubRshort(itsRshort), mElTubRlong(itsRlong)
   {
   
    pMessenger = new GateElTubMessenger(this);
    
   }
//--------------------------------------------------------------------------------- 


//---------------------------------------------------------------------------------
   GateElTub::GateElTub(const G4String& itsName,
		                            G4bool itsFlagAcceptChildren,
			                    G4int depth)				      
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    pElTubSolid(0), pElTubLog(0)
   {
    mElTubHeight = 1.0*cm; 
    mElTubRshort = 1.0*cm;
    mElTubRlong = 1.5*cm;
    pMessenger = new GateElTubMessenger(this);
   }
//--------------------------------------------------------------------------------- 


//---------------------------------------------------------------------------------
//Destuctor
//---------------------------------------------------------------------------------
   GateElTub::~GateElTub()
   {
     
     delete pMessenger;
     
   }
//--------------------------------------------------------------------------------- 


//---------------------------------------------------------------------------------
G4LogicalVolume* GateElTub::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  
  if (!flagUpdateOnly || !pElTubSolid) {
    // Build mode: build the solid, then the logical volume
    pElTubSolid
      = new G4EllipticalTube(GetSolidName(), 
      		   mElTubRshort, 
	  	   mElTubRlong,
		   mElTubHeight/2.);
    pElTubLog
      = new G4LogicalVolume(pElTubSolid, mater, GetLogicalVolumeName(),0,0,0);

  } 
  else {
    // Update mode: refresh the dimensions of the solid
	    pElTubSolid->SetDz(GetElTubHalfHeight());
	    pElTubSolid->SetDx(GetElTubRshort());
	    pElTubSolid->SetDy(GetElTubRlong());
  }
  
  return pElTubLog;
 
}
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
void GateElTub::DestroyOwnSolidAndLogicalVolume()
{
    
  if (pElTubLog)
    delete pElTubLog;
    pElTubLog = 0;

  if (pElTubSolid)
    delete pElTubSolid;
    pElTubSolid = 0;

}
//---------------------------------------------------------------------------------
