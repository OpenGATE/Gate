/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GATEAnnihilationRangeACTOR_HH
#define GATEAnnihilationRangeACTOR_HH

#include "GateConfiguration.h"
#include "GateVActor.hh"
#include "GateAnnihilationRangeActorMessenger.hh"
#include "GateTreeFileManager.hh"

//-----------------------------------------------------------------------------
class GateAnnihilationRangeActor : public GateVActor {
public:

    virtual ~GateAnnihilationRangeActor();

    // This macro initialize the CreatePrototype and CreateInstance
    FCT_FOR_AUTO_CREATOR_ACTOR(GateAnnihilationRangeActor)

    // Constructs the sensor
    virtual void Construct();

    // Callbacks
    virtual void PostUserTrackingAction(const GateVVolume *, const G4Track *);

    /// Saves the data collected to the file
    virtual void SaveData();

    virtual void ResetData();


protected:
    GateAnnihilationRangeActor(G4String name, G4int depth = 0);

    GateOutputTreeFileManager *mFile;
    double mX;
    double mY;
    double mZ;

    GateAnnihilationRangeActorMessenger *pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(AnnihilationRangeActor, GateAnnihilationRangeActor)

#endif /* end #define GATEAnnihilationRangeACTOR_HH */
