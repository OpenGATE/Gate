/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDigitizerMgrMessenger_h
#define GateDigitizerMgrMessenger_h 1

#include "GateDigitizerMgr.hh"
#include "G4UImessenger.hh"
#include "globals.hh"
#include "GateClockDependentMessenger.hh"
//#include "GateSinglesDigitizer.hh"

class GateDigitizerMgr;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

//public GateClockDependentMessenger


//InsertionBase --> Collection

class GateDigitizerMgrMessenger: public GateClockDependentMessenger
{
public:
  GateDigitizerMgrMessenger(GateDigitizerMgr*);
  ~GateDigitizerMgrMessenger();

  void SetNewValue(G4UIcommand*, G4String);

  //! Get the current value of the insertion name
  inline const G4String& GetNewCollectionName()
    { return m_newCollectionName; }

  //! Set the value of the insertion name
  inline void SetNewCollectionName(const G4String& val)
    { m_newCollectionName = val; }

  //! Check whether there is a name conflict between a new
  //! attachment and an already existing one
  virtual G4bool CheckNameConflict(const G4String& name);

  //! Check and solves name conflict between a new
  //! attachment and already existing ones
  virtual void AvoidNameConflicts();


private:
  //! Dumps the list of modules that the user can insert into the list
  virtual const G4String& DumpMap();

   //! Lists all the system-names onto the standard output
  virtual void ListChoices()
    { G4cout << "The available choices are: " << DumpMap() << "\n"; }

 //! Inserts a new module into the pulse-processor list
  virtual void DoInsertion(const G4String& typeName);

  //! Get the chain list
  inline GateDigitizerMgr* GetDigitizerMgr()
    { return (GateDigitizerMgr*) GetClockDependent(); }

private:

  G4UIcmdWithAString*         DefineNameCmd;	      //!< the UI command 'name'
  G4UIcmdWithoutParameter*    ListChoicesCmd;       //!< the UI command 'info'
  G4UIcmdWithoutParameter*    ListCmd;	      //!< the UI command 'list'
  G4UIcmdWithAString*         pInsertCmd;	      //!< the UI command 'insert'
  G4UIcmdWithAString* 		  SetChooseSDCmd;     //!< the UI command 'chooseSD'
private:
  G4String  	      	m_newCollectionName; //m_newInsertionBaseName  //!< the name to be given to the next insertion
  	  	  	  	  	  	  	  	  	  	  	  	  //!< (if empty, the type-name will be used)

  G4String m_SDname;

};

#endif
