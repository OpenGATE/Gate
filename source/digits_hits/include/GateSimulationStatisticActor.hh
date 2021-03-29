/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class GateSimulationStatisticActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#ifndef GATESIMULATIONSTATISTICACTOR_HH
#define GATESIMULATIONSTATISTICACTOR_HH

#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateActorMessenger.hh"
#include "GateSimulationStatisticActorMessenger.hh"

#include <sys/time.h>

//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateSimulationStatisticActor : public GateVActor {
public:

    virtual ~GateSimulationStatisticActor();

    //-----------------------------------------------------------------------------
    // This macro initialize the CreatePrototype and CreateInstance
    FCT_FOR_AUTO_CREATOR_ACTOR(GateSimulationStatisticActor)

    //-----------------------------------------------------------------------------
    // Constructs the sensor
    virtual void Construct();

    // Options
    void SetTrackTypesFlag(bool b) { mTrackTypesFlag = b; }

    //-----------------------------------------------------------------------------
    // Callbacks
    virtual void BeginOfRunAction(const G4Run *);

    virtual void BeginOfEventAction(const G4Event *);

    virtual void PreUserTrackingAction(const GateVVolume *, const G4Track *);

    virtual void UserSteppingAction(const GateVVolume *, const G4Step *);

    //-----------------------------------------------------------------------------
    /// Saves the data collected to the file
    virtual void SaveData();

    virtual void ResetData();

protected:
    GateSimulationStatisticActor(G4String name, G4int depth = 0);

    long int mNumberOfRuns;
    long int mNumberOfEvents;
    long int mNumberOfTrack;
    long long int mNumberOfSteps;
    long long int mNumberOfGeometricalSteps;
    long long int mNumberOfPhysicalSteps;
    timeval start;
    timeval start_afterinit;
    std::string startDateStr;

    bool mTrackTypesFlag;
    std::map<std::string, int> mTrackTypes;

    GateSimulationStatisticActorMessenger *pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(SimulationStatisticActor, GateSimulationStatisticActor)


#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
