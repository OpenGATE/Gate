/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEREGULARPARAMETERIZEDINSERTERMESSENGER_HH
#define GATEREGULARPARAMETERIZEDINSERTERMESSENGER_HH 1

#include "globals.hh"
#include "GateMessenger.hh"

class GateRegularParameterized;

class GateRegularParameterizedMessenger : public GateMessenger
{
public:

    //! Constructor
    GateRegularParameterizedMessenger(GateRegularParameterized* itsInserter);

    //! Destructor
    ~GateRegularParameterizedMessenger();

    //! SetNewValue
    void SetNewValue(G4UIcommand*, G4String);

    //! Get the RegularParameterizedInserter
    virtual inline GateRegularParameterized* GetRegularParameterizedInserter()
      { return m_inserter; }

private:

    G4UIcmdWithoutParameter*        AttachPhantomSDCmd;
    G4UIcmdWithAString*             InsertReaderCmd;
    G4UIcmdWithoutParameter*        RemoveReaderCmd;
    G4UIcmdWithAString*             AddOutputCmd;
    G4UIcmdWithABool*               SkipEqualMaterialsCmd;
    G4UIcmdWithAnInteger*           VerboseCmd;

    GateRegularParameterized*  m_inserter;
};

#endif
