/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateSystemListManager_h
#define GateSystemListManager_h 1

#include "globals.hh"

#include "GateListManager.hh"

class GateVSystem;
class GateSystemListMessenger;
class GateSystemComponent;
class GateVVolume;


/*! \class  GateSystemListManager
    \brief  Stores the list of systems
    
    - GateSystemListManager - by Daniel.Strul@iphe.unil.ch (May 16 2002)
    
    - The GateSystemListManager is a singleton. Its task is to handle a list of systems,
      provide tools for retrieving systems ort system-components, and allow the
      insertion of new systems.
    
      \sa GateVSystem
*/      
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.
//    Last modification in 07/2012 by vesna.cuplov@gmail.com, for the OPTICAL system.

class GateSystemListManager : public GateListManager
{
  public:
    /*! This function allows to retrieve the current instance of the GateSystemListManager singleton

      	If the GateSystemListManager already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateSystemListManager constructor
    */
    static GateSystemListManager* GetInstance(); 

    virtual ~GateSystemListManager(); //!< Public destructor

  private:
    GateSystemListManager();   //!< Private constructor: this function should only be called from GetInstance()


  public:
    //! \name Methods to manage the listgeometry/include/GateOPETSystem.hh:#ifndef
    //@{
    virtual void RegisterSystem(GateVSystem* newSystem);   //!< Registers a new system in the list
    virtual void UnregisterSystem(GateVSystem* system);    //!< Removes a deleted system from the system-list
    virtual void ListSystems() const  	      	      	   //!< List the systems stored in the list
      { TheListElements();}
      //{ TheListElements();}
    //@}
        
    //! \name Methods to retrieve a system
    //@{

    //! Retrieves a system from its name
    inline GateVSystem* FindSystem(const G4String& name)        	      
      { return (GateVSystem*) FindElement(name); }

    /*! Tries to find to which system an inserter is attached
      	(either directly or through one of its ancestors)

      	\param anCreator the inserter to test
  
      	\return the system to which the inserter is attached, if any
    */
    GateVSystem* FindSystemOfCreator(GateVVolume* anCreator);

    //@}

    //! \name Access methods
    //@{
    GateVSystem* GetSystem(size_t i) 
      {return (GateVSystem*) GetElement(i);}      	     //!< Retrieves a system from a store-iterator
    //@}

    //! \name Methods to create new systems
    //@{

    /*! Check whether a new inserter has the same name as one of the predefined systems
      	If that's the case, auto-create the system

      	\param newChildCreator the newly-created inserter  
    */
    void CheckScannerAutoCreation(GateVVolume* newChildCreator);

    /*! Checks whether a name corresponds to onw of the predefined system-names
    
      	\param name the name to check
	
	\return the position of the name in the name-table (-1 if not found)
    */
    G4int DecodeTypeName(const G4String& name);

    /*! Create a new system of a specific type
      	
	\param childTypeName: the type-name of the system to create
	
	\return the newly created system
    */
    virtual GateVSystem* InsertNewSystem(const G4String& typeName);

    //@}

    //! Lists all the system-names into a string
    virtual const G4String& DumpChoices();

    //! Lists all the system-names onto the standard output
    virtual void ListChoices() 
      { G4cout << "The available choices are: " << DumpChoices() << "\n"; }

    // Get the list of inserted systems names
    inline std::vector<G4String>* GetInsertedSystemsNames() const {return theInsertedSystemsNames;}

    inline G4bool GetIsAnySystemDefined(){return m_isAnySystemDefined;}
    inline void SetIsAnySystemDefined(G4bool val){m_isAnySystemDefined=val;}

  protected:
    GateSystemListMessenger* m_messenger;    //!< Pointer to the store's messenger

  private:
    //! Pointer to the systems names.
    std::vector<G4String>* theInsertedSystemsNames;
    //! Static pointer to the GateSystemListManager singleton
    static GateSystemListManager* theGateSystemListManager;

    static const G4String     	theSystemNameList[];  //!< the list of predefined-system names
    //OK GND 2023
    G4bool m_isAnySystemDefined;


};


#endif

