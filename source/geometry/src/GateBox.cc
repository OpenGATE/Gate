/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBox.hh"
#include "GateBoxMessenger.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"

#include "G4ThreeVector.hh"

#include "globals.hh"

//---------------------------------------------------------------------------------
// Initialisation
// Constructor
//------------------------------------------------------------------------------------------------
GateBox::GateBox(const G4String& itsName, const G4String& itsMaterialName, //DS comment to avoid warning
      	      	 G4double itsXLength, G4double itsYLength, G4double itsZLength,
		 G4bool itsFlagAcceptChildren, G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    pBoxSolid(0), pBoxLog(0), pBoxPhys(0), pMessenger(0) 
{
  SetMaterialName(itsMaterialName);
  mBoxLength[0] = itsXLength;
  mBoxLength[1] = itsYLength;
  mBoxLength[2] = itsZLength;
     
  pMessenger = new GateBoxMessenger(this);
}
//------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------
GateBox::GateBox(const G4String& itsName,
		 G4bool itsFlagAcceptChildren,
		 G4int depth)
  : GateVVolume(itsName, itsFlagAcceptChildren, depth),
    pBoxSolid(0), pBoxLog(0), pBoxPhys(0), pMessenger(0) 
{
     
  // Set default material name
  SetMaterialName("Vacuum");
  mBoxLength[0] = 1.0*cm;
  mBoxLength[1] = 1.0*cm;
  mBoxLength[2] = 1.0*cm;
  
  pMessenger = new GateBoxMessenger(this);
}
//---------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
//Destructor
GateBox::~GateBox()
{
         
  delete pMessenger;
     
}
//------------------------------------------------------------------------------------------------   
   
//------------------------------------------------------------------------------------------------
G4LogicalVolume* GateBox::ConstructOwnSolidAndLogicalVolume(G4Material* mater, G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly || !pBoxSolid){      
    // Solid
    pBoxSolid = new G4Box(GetSolidName(), GetBoxXHalfLength(), GetBoxYHalfLength(), GetBoxZHalfLength());
  
   
    // Logical volume 
    pBoxLog = new G4LogicalVolume(pBoxSolid, mater, GetLogicalVolumeName()); 
  
  }
  else if (pBoxSolid)
    {
   
      pBoxSolid->SetXHalfLength(GetBoxXHalfLength());
      pBoxSolid->SetYHalfLength(GetBoxYHalfLength());
      pBoxSolid->SetZHalfLength(GetBoxZHalfLength());    

    }
  
  return pBoxLog;
}  
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
// Sets X Box length 
void GateBox::SetBoxXLength(G4double lengthXChoice)
{
  mBoxLength[0] = lengthXChoice;
} 
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// Sets Y Box length 
void GateBox::SetBoxYLength(G4double lengthYChoice)
{
  mBoxLength[1] = lengthYChoice;
}
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// Sets Z Box length 
void GateBox::SetBoxZLength(G4double lengthZChoice)
{ 
  mBoxLength[2] = lengthZChoice;
}
//------------------------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------------------------
void GateBox::DestroyOwnSolidAndLogicalVolume()
{
  if (pBoxLog){
    delete pBoxLog;}

  pBoxLog = 0;
    
  if (pBoxSolid){
    delete pBoxSolid;}    

  pBoxSolid = 0;     
}  
//------------------------------------------------------------------------------------------------
