/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateObjectStore.hh"

#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"


//----------------------------------------------------------------------------
// Static pointer to the GateObjectStore singleton
GateObjectStore* GateObjectStore::theGateObjectStore=0;



/*    	This function allows to retrieve the current instance of the GateObjectStore singleton
      	If the GateObjectStore already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateObjectStore constructor
*/
GateObjectStore* GateObjectStore::GetInstance()
{
  if (!theGateObjectStore)
    theGateObjectStore = new GateObjectStore();
  return theGateObjectStore;
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// Private constructor
GateObjectStore::GateObjectStore()
{  
  ;
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// Public destructor
GateObjectStore::~GateObjectStore()
{  
  ;
}
//----------------------------------------------------------------------------



//----------------------------------------------------------------------------
// Registers a new object-creator in the creator list
void GateObjectStore::RegisterCreator(GateVVolume* newCreator)
{   
  insert(MapPair(newCreator->GetObjectName(),newCreator));
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// Removes a deleted object-creator from the creator-list    
void GateObjectStore::UnregisterCreator(GateVVolume* creator) 
{
  erase(creator->GetObjectName());
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// List the object creators stored in the creator list
void GateObjectStore::ListCreators()
{
  G4cout << "Nb of volumes:       " << size() << G4endl;
  G4cout << DumpMap(true,"","\n","\t\t") << G4endl;
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// Retrieves a creator
GateVVolume* GateObjectStore::FindCreator(const G4String& name)
{
  iterator iter = find(name);
  return (iter==end()) ? 0 : GetCreator(iter);
}
//----------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Retrieves the inserter of a volume by his name
GateVVolume* GateObjectStore::FindVolumeCreator(const G4String& volumeName)
{  
  // for (iterator iter = begin() ; iter != end() ; ++iter){
  //     DD(GetCreator(iter)->GetObjectName());
  //     DD(GetCreator(iter));
  //     DD(GetCreator(iter)->GetLogicalVolume());
  //   }
  G4String list;
  for (iterator iter = begin() ; iter != end() ; ++iter){
    list += " "+ GetCreator(iter)->GetObjectName();
    if ( GetCreator(iter)->GetObjectName() == volumeName) return GetCreator(iter);
  }

  GateError("The volume '" << volumeName << "' cannot be found in the list of volumes. Abort. "
            << G4endl << "Here is the list of available volumes : " 
            << G4endl << list << G4endl);
  return 0;
}
//---------------------------------------------------------------------------



//---------------------------------------------------------------------------
// Retrieves the inserter of a volume by his name
GateVVolume* GateObjectStore::FindVolumeCreator(G4VPhysicalVolume* volume)
{
   
  for (iterator iter = begin() ; iter != end() ; ++iter){
        
  if ( GetCreator(iter)->GetPhysicalVolumeName() == volume->GetName()){
       
      return GetCreator(iter);}
   }   
  return 0;
}
//---------------------------------------------------------------------------



//----------------------------------------------------------------------------
// Retrieves the creator of a logical volume
/*
GateVVolume* GateObjectStore::FindVolumeCreator(G4VPhysicalVolume* volume)
{
  return FindVolumeCreator(volume->GetName());
}
*/
//----------------------------------------------------------------------------


