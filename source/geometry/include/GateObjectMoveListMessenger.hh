/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateObjectMoveListMessenger_h
#define GateObjectMoveListMessenger_h 1

#include "globals.hh"
#include "GateListMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;


#include "GateObjectRepeaterList.hh"

class GateVVolume;

/*! \class GateObjectMoveListMessenger
  \brief A messenger for inserting movement objects (i.e. instances of application
  \brief classes derived from GateVObjectMove)

  - GateObjectMoveListMessenger - by Daniel.Strul@iphe.unil.ch

  - The GateObjectMoveListMessenger inherits from the abilities/responsabilities
  of the GateListMessenger base-class: management of an object
  attachment list; creation and management of a Gate UI directory for
  this attachment list; creation of the UI commands
  "name", "insert", "info" and "list", ability to compute new insertion names
  and to avoid name conflicts

  - In addition, it implements the pure virtual method DumpMap() that stores
  and returns the list of insertion type names; it overloads the method
  ListChoices() to provide more information on the various choices available;
  it implements the pure virtual method DoInsertion() that is responsible
  for performing the insertions

*/
class GateObjectMoveListMessenger: public GateListMessenger
{
public:
  //! Constructor
  GateObjectMoveListMessenger(GateObjectRepeaterList* itsRepeaterList);
  //! Destructor
  ~GateObjectMoveListMessenger();

  //! Command interpreter
  void SetNewValue(G4UIcommand*, G4String);

  virtual inline void SetFlagMove(G4bool val)  { moveFlag = val; G4cout << " MoveMess::SetFlagMove " << val << G4endl; };
  virtual inline G4bool GetFlagMove() const { return moveFlag; };

  //e    virtual inline void SetObjectMoveListMess(GateObjectMoveListMessenger* moveMess) { moveListMess = moveMess; };
  //e    virtual inline GateObjectMoveListMessenger* GetObjectMoveListMess() { return moveListMess; };

private:
  //! Implementation of the pure virtual method DumpMap() defined by the super-class GateListMessenger
  //! This method stores and returns the list of insertion type names
  virtual const G4String& DumpMap();

  //! Overloading of the virtual method ListChoices() defined by the super-class GateListMessenger
  //! This method prints out some further guidance on the various choices available
  virtual void ListChoices();

  //! Implementation of the pure virtual method DoInsertion() defined by the super-class GateListMessenger
  //! This method create and inserts a new move of the type typeName
  virtual void DoInsertion(const G4String& typeName);

  //! Returns the repeater list controlled by this messenger
  virtual inline GateObjectRepeaterList* GetRepeaterList()
  { return (GateObjectRepeaterList*) GetListManager();}

  //! Returns the inserter whose repeater list is controlled by this messenger
  virtual GateVVolume* GetCreator()
  { return GetRepeaterList()->GetCreator(); }

private :
  G4bool moveFlag;
};

#endif
