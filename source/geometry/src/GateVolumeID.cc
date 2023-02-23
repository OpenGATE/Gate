/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVolumeID.hh"

#include "G4UnitsTable.hh"
#include "G4TouchableHistory.hh"

#include "GateDetectorConstruction.hh"
#include "GateObjectStore.hh"
#include "GateObjectChildList.hh"


//-----------------------------------------------------------------------------------
// Constructs a GateVolumeSelector for a physical volume
GateVolumeSelector::GateVolumeSelector(G4VPhysicalVolume* itsVolume)
{

  m_creator = GateObjectStore::GetInstance()->FindVolumeCreator(itsVolume);
  
  m_copyNo = itsVolume->GetCopyNo();
    
  if (m_creator->GetMotherList()){ 
    m_daughterID = m_creator->GetMotherList()->GetChildNo(m_creator,m_copyNo);
  }  
  else{
    m_daughterID = 0;}
  
}
//-----------------------------------------------------------------------------------   


//-----------------------------------------------------------------------------------
// Friend function: inserts (prints) a GateVolumeSelector into a stream
std::ostream& operator<<(std::ostream& flux, const GateVolumeSelector& volumeLevelID)    
{
    flux << volumeLevelID.GetCreator()->GetObjectName() << "-" << volumeLevelID.GetCopyNo() << "/" << volumeLevelID.GetVolume()->GetName() << "-" << volumeLevelID.GetDaughterID() ;
    return flux;
}
//-----------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------
// Computes a new GateVolumeID for a touchable,
GateVolumeID::GateVolumeID(const G4TouchableHistory* touchable)
{ 	
   
  // Get the current physical volume
  if ( !touchable) {
      G4cout << "[GateVolumeID::GateVolumeID]: The touchable is null!\n";
      return;
  }

   G4VPhysicalVolume* physVol = touchable->GetVolume();
  if ( !physVol) {
      G4cout << "[GateVolumeID::GateVolumeID]: The volume is null!\n";
      return;
      
  G4cout << " FIN Constructeur GateVolumeID\n";    
  }

/*
  InsertVolumeLevel( physVol );

  // We climb the geometry until we reach the world volume
  while (physVol->GetMother())
  {
    physVol = physVol->GetMother();
    InsertVolumeLevel( physVol );
  }
*/

//   replacement with a GEANT4.6 compatible code:
  for (G4int numVol=0;numVol<touchable->GetHistoryDepth();numVol++){
     
    InsertVolumeLevel( touchable->GetVolume(numVol) );
    
  }
    
  physVol = GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume();
  InsertVolumeLevel( physVol );
  
}
//-----------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
// Recreate a volumeID from data stored in a hit file
GateVolumeID::GateVolumeID(G4int *daughterID, size_t arraySize)
{ 	
  G4VPhysicalVolume* physVol = GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume();
  push_back( GateVolumeSelector(physVol) );
  
  for (size_t pos=1; pos<arraySize ; pos++){
    if (daughterID[pos]>=0)  {
      physVol = physVol->GetLogicalVolume()->GetDaughter(daughterID[pos]);
      push_back( GateVolumeSelector(physVol) );
    } else {
      break;
    }
  }
}
//-----------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------
G4int GateVolumeID::GetCreatorDepth (G4String name) const                
{     
  G4int ctrl = -1;
  size_t depth;
  for (depth = 0 ; depth < size() ; depth ++)
    if (GetCreator(depth)->GetObjectName() == name) {
      ctrl = 0;
      break;
    }
  if (ctrl == 0)
    return depth;
  else
    return ctrl;
}
//-----------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------
// Retrieves the affine transformation connecting connecting the bottom volume to one of its ancestors
// The method parameter is the ancestor depth, i.e. the position of the ancestor in the vector (0->world volume)
G4AffineTransform GateVolumeID::ComputeAffineTransform(G4int ancestorDepth) const
{
  // Retrieve the ancestor's depth
  if ( !IsValidDepth(ancestorDepth) ) {
      G4cout << "[GateVolumeID::ComputeAffineTransform]: level "<< ancestorDepth << " is out-of-range!!!\n";
      return G4AffineTransform();
  }

  // Compute the affine tranform as the product of all the affine transforms of the volumes located
  // between the current depth (ancestor's depth) and the bottom volume depth
  G4AffineTransform targetTransform;
  for ( size_t i=ancestorDepth+1 ; i<size(); i++) 
      targetTransform = GetVolumeAffineTransform(GetVolume(i)) * targetTransform;

  // Return the final product
  return targetTransform;
}
//-----------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------
// Tool function: returns the affine transform linking a volume's reference frame to its mother's reference frame
G4AffineTransform GateVolumeID::GetVolumeAffineTransform(G4VPhysicalVolume* physicalVolume) 
{   
    if (!physicalVolume) {
      G4cout << "[GateVolumeID::GetVolumeAffineTransform]: volume is null!!!\n";
      return G4AffineTransform();
    }
    return G4AffineTransform(physicalVolume->GetRotation(),physicalVolume->GetTranslation()); 
}
//-----------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------
 /* Move a position from the reference frame of the bottom-volume into the reference frame of one of its ancestors 
  Params:
	position:     the position to be transfered into the ancestor's reference frame
	ancestorLevel:     The level of an ancestor of the bottom-volume, located somewhere between the top and the bottom.
	      	      	   If no level is specified, the world-level is selected
*/    
G4ThreeVector GateVolumeID::MoveToAncestorVolumeFrame(G4ThreeVector position,G4int ancestorLevel) const
{
  G4AffineTransform aTransform = ComputeAffineTransform(ancestorLevel);
  return MoveToFrameByTransform(position,aTransform,true);
}
//-----------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------
/* Move a position from the reference frame of a bottom-volume's ancestor into the reference frame of the bottom-volume
  Params:
	position:     a position to be transfered into another reference frame
	ancestorLevel:     The level of an ancestor of the bottom-volume, located somewhere between the top and the bottom.
	      	      	   If no level is specified, the world-level is selected
*/    
G4ThreeVector GateVolumeID::MoveToBottomVolumeFrame(G4ThreeVector position,G4int ancestorLevel) const
{
  G4AffineTransform transform = ComputeAffineTransform(ancestorLevel);
  return MoveToFrameByTransform(position,transform,false);
}
//-----------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------
/* Move a position from one reference frame into another 
  Params:
	position:  a position to be transfered into another reference frame
	transform: the transformation linking the two reference frames
	flagDirect: if true, apply the transform. If false, apply its inverse.
*/    
G4ThreeVector GateVolumeID::MoveToFrameByTransform(G4ThreeVector position,const G4AffineTransform& transform,G4bool flagDirect) const
{
  if (flagDirect)
    transform.ApplyPointTransform(position);
  else
    transform.Inverse().ApplyPointTransform(position);
  return position;
}
//-----------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------
// Friend function: inserts (prints) a GateVolumeID into a stream
std::ostream& operator<<(std::ostream& flux, const GateVolumeID& volumeID)    
{
    flux    << "Vol(0" ;
    for (size_t i=0; i<volumeID.size(); i++) 
      	    flux << " -> " << volumeID[i] ;
    flux    << ")";
    
    return flux;
}
//-----------------------------------------------------------------------------------


