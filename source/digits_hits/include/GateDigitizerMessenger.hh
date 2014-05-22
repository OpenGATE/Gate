/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDigitizerMessenger_h
#define GateDigitizerMessenger_h 1

#include "GateClockDependentMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

#include "GateDigitizer.hh"



/*! \class GateDigitizerMessenger
    \brief This messenger manages a list of pulse-processor chains

    - GateDigitizerMessenger - by Daniel.Strul@iphe.unil.ch
*/
class GateDigitizerMessenger: public GateClockDependentMessenger
{
  public:
    GateDigitizerMessenger(GateDigitizer* itsDigitizer);  //!< constructor
    ~GateDigitizerMessenger();                            //!< destructor

     //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Get the current value of the insertion name
    inline const G4String& GetNewInsertionBaseName()
      { return m_newInsertionBaseName; }

    //! Set the value of the insertion name
    inline void SetNewInsertionBaseName(const G4String& val)
      { m_newInsertionBaseName = val; }

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
    inline GateDigitizer* GetDigitizer()
      { return (GateDigitizer*) GetClockDependent(); }

  private:

    G4UIcmdWithAString*         DefineNameCmd;	      //!< the UI command 'name'
    G4UIcmdWithoutParameter*    ListChoicesCmd;       //!< the UI command 'info'
    G4UIcmdWithoutParameter*    ListCmd;	      //!< the UI command 'list'
    G4UIcmdWithAString*         pInsertCmd;	      //!< the UI command 'insert'

  private:
    G4String  	      	m_newInsertionBaseName;  //!< the name to be given to the next insertion
      	      	      	                         //!< (if empty, the type-name will be used)
};

#endif
