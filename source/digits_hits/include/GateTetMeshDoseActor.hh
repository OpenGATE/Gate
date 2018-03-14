/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GATE_TETMESH_DOSE_ACTOR_HH
#define GATE_TETMESH_DOSE_ACTOR_HH 

#include <memory>
#include <map>

#include <G4Types.hh>
#include <G4String.hh>
#include <G4THitsMap.hh>
#include <G4HCofThisEvent.hh>
#include <G4Step.hh>
#include <G4Run.hh>
#include <G4Event.hh>

#include "GateVVolume.hh"

#include "GateVActor.hh"

class GateActorMessenger;


class GateTetMeshDoseActor : public GateVActor
{
  public:
    struct Estimators
    {
      G4double dose;
      G4double sumOfSquaredDose;
      G4double relativeUncertainty;
    };

    void Construct() final;
    
    FCT_FOR_AUTO_CREATOR_ACTOR(GateTetMeshDoseActor)

    // user callbacks for this actor, will be invoked in 'GateUserActions'
    void BeginOfRunAction(const G4Run*) final;
    void EndOfRunAction(const G4Run*) final;
    void EndOfEventAction(const G4Event*) final;
    
    // Will be called before first run
    void InitData();
    
    // pure virtual from GateVActor:
    //  - how to save data to disk
    //  - how to reset data in RAM
    void SaveData() final;
    void ResetData() final;

    // Implements the virtual functions of G4PrimitivScorer.
    void Initialize(G4HCofThisEvent*) final;
    void EndOfEvent(G4HCofThisEvent*) final;
    void clear() final;

    // Is linked to 'ProcessHits' in GateVActor's interface. Omits the
    // usage of a read-out geometry.
    void UserSteppingAction(const GateVVolume*, const G4Step*) final;
  
  protected:
    GateTetMeshDoseActor(G4String name, G4int depth = 0);

  private:
    // key = copy number of a tetrahedron's physical volume
    // value = deposited dose
    std::map<G4int, G4double> mEvtDoseMap;

    G4int mRunCounter;

    // one entry per tetrahedron
    std::vector<Estimators> mRunData;

    std::unique_ptr<GateActorMessenger> pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(TetMeshDoseActor,GateTetMeshDoseActor)

#endif  // GATE_TETMESH_DOSE_ACTOR_HH
