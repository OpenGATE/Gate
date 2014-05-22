/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateObjectStore_h
#define GateObjectStore_h 1

#include "globals.hh"
#include <vector>

#include "GateVVolume.hh"
#include "GateMaps.hh"

class G4LogicalVolume;

/*! \class  GateObjectStore
    \brief  Stores the list of object creators: allows to retrieve creators/volumes
    
    - GateObjectStore - by Daniel.Strul@iphe.unil.ch (May 9 2002)
    
    - The GateObjectStore is a singleton. Its task is to provide tools for listing 
      or retrieving object-creators 
    
      \sa GateVVolume
*/      
class GateObjectStore : public GateMap<G4String,GateVVolume*>
{
  public:
    //! This function allows to retrieve the current instance of the GateObjectStore singleton
    /*! 
      	If the GateObjectStore already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateObjectStore constructor
    */
    static GateObjectStore* GetInstance(); 

    virtual ~GateObjectStore(); //!< Public destructor

  private:
    GateObjectStore();   //!< Private constructor: this function should only be called from GetInstance()


  public:
    //! \name Methods to manage the creator list
    //@{
    virtual void RegisterCreator(GateVVolume* newCreator);   //!< Registers a new object-creator in the creator list
    virtual void UnregisterCreator(GateVVolume* creator);    //!< Removes a deleted object-creator from the creator-list
    virtual void ListCreators();      	      	      	      	    //!< List the object creators stored in the creator list
    //@}
        
    //! \name Methods to retrieve a creator
    //@{
    GateVVolume* FindCreator(const G4String& name);                        //!< Retrieves a creator
    GateVVolume* FindVolumeCreator(const G4String& volumeName);      	      	
    GateVVolume* FindVolumeCreator(G4VPhysicalVolume* volume);       	//!< Retrieves the creator of a logical volume
    //@}

    //! \name Methods for accessing items of the store
    //@{
    const G4String& GetCreatorName(iterator iter) {return iter->first;}       //!< Retrieves a creator name from a store-iterator
    GateVVolume* GetCreator(iterator iter) {return iter->second;}      //!< Retrieves a creator from a store-iterator
    //@}

  private:
    //! Static pointer to the GateObjectStore singleton
    static GateObjectStore* theGateObjectStore;
};


#endif

