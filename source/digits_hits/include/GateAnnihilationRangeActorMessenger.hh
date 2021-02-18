/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GATEAnnihilationRangeACTORMESSENGER_HH
#define GATEAnnihilationRangeACTORMESSENGER_HH

#include "GateConfiguration.h"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateAnnihilationRangeActor;

//-----------------------------------------------------------------------------
class GateAnnihilationRangeActorMessenger : public GateActorMessenger {
public:

    //-----------------------------------------------------------------------------
    // Constructor with pointer on the associated sensor
    GateAnnihilationRangeActorMessenger(GateAnnihilationRangeActor *v);

    // Destructor
    virtual ~GateAnnihilationRangeActorMessenger();

    /// Command processing callback
    virtual void SetNewValue(G4UIcommand *, G4String);

    void BuildCommands(G4String base);

protected:

    /// Associated sensor
    GateAnnihilationRangeActor *pActor;

    /// Command objects
    G4UIcmdWithADoubleAndUnit *pLmaxCmd;
    G4UIcmdWithADoubleAndUnit *pLminCmd;
    G4UIcmdWithAnInteger *pNBinsCmd;

}; // end class GateAnnihilationRangeActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEAnnihilationRangeACTORMESSENGER_HH */