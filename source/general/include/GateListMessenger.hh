/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateListMessenger_h
#define GateListMessenger_h 1

#include "GateClockDependentMessenger.hh"
#include "GateListManager.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;

/*! \class GateListMessenger
    \brief Abstract base class for managing an attachment list (GateListManager) and for inserting new objects
    
    - GateListMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateListMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class, i.e. the creation and management
      of a Gate UI directory for a Gate object
      
    - In addition, the main responsability of this messenger is to handle
      an attachment list, and to allow the insertion of new attachments.

    - It proposes and manages commands specific to the attachment lists
      definition of the name of a new attachment, listing of possible choices,
      list of already created attachments, and insertion of a new attachment
      
    - The class contains two pure virtual methods: DumpMap() and DoInsertion()
      These methods must be implemented in derived concrete classes.

*/      
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateListMessenger: public GateClockDependentMessenger
{
  public:
    //! constructor
    GateListMessenger(GateListManager* itsListManager);

    //! destructor
    virtual ~GateListMessenger();
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Get a pointer to the list manager
    inline GateListManager* GetListManager() 
      { return (GateListManager*) GetClockDependent(); }

    //! Get the current value of the insertion name
    inline const G4String& GetNewInsertionBaseName() 
      { return mNewInsertionBaseName; }

    //! Setter and getter for the system type.
    inline void SetSystemType(const G4String& val) { mSystemType = val; }
    inline G4String GetSystemType() const { return mSystemType; }
    
    //! Set the value of the insertion name
    inline void SetNewInsertionBaseName(const G4String& val) 
      { mNewInsertionBaseName = val; }
      
    //! Check whether there is a name conflict between a new
    //! attachment and an already existing one
    virtual G4bool CheckNameConflict(const G4String& name);

    //! Check and solves name conflict between a new
    //! attachment and already existing ones
    virtual void AvoidNameConflicts();


  private:
    //! Pure virtual method: lists all the attachment-names into a string
    virtual const G4String& DumpMap() =0;

    //! Lists all the system-names onto the standard output
    virtual void ListChoices() 
      { G4cout << "The available choices are: " << DumpMap() << "\n"; }

    //! Pure virtual method: create and insert a new attachment
    virtual void DoInsertion(const G4String& typeName)=0;

  protected:

    G4UIcmdWithAString*         pSystemTypeCmd;       //!< the UI command 'systemType'
    G4UIcmdWithAString*         pDefineNameCmd;	      //!< the UI command 'name'
    G4UIcmdWithoutParameter*    pListChoicesCmd;       //!< the UI command 'info'
    G4UIcmdWithoutParameter*    pListCmd;	      //!< the UI command 'list'
    G4UIcmdWithAString*       	pInsertCmd;	      //!< the UI command 'insert'
    
  private:
    static G4String  	      	mSystemType;    //!< carries the system type.
    G4String  	      	mNewInsertionBaseName;  //!< the name to be given to the next insertion
      	      	      	      	      	         //!< (if empty, the type-name will be used)
};

#endif

