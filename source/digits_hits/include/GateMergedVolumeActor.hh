/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class GateMergedVolumeActor
  \author Didier Benoit (didier.benoit@inserm.fr)
 */

#ifndef GATEMERGEDVOLUMEACTOR_HH
#define GATEMERGEDVOLUMEACTOR_HH

#include "GateVActor.hh"
class GateMergedVolumeActorMessenger;
class GateActorMessenger;

//-----------------------------------------------------------------------------
class GateMergedVolumeActor : public GateVActor
{
  public:

    virtual ~GateMergedVolumeActor();

    //-----------------------------------------------------------------------------
    // This macro initialize the CreatePrototype and CreateInstance
    FCT_FOR_AUTO_CREATOR_ACTOR(GateMergedVolumeActor)

    //-----------------------------------------------------------------------------
    // Constructs the sensor
    virtual void Construct();
    //virtual void PostUserTrackingAction(const G4Track*);
    //virtual void UserSteppingAction(const GateVVolume *, const G4Step*){}
    virtual void clear(){ ResetData(); }
    virtual void ResetData() {}
    virtual G4bool ProcessHits(G4Step * step , G4TouchableHistory* th);
    virtual void Initialize(G4HCofThisEvent*){}
    virtual void EndOfEvent(G4HCofThisEvent*){}

    void ListOfVolumesToMerge( G4String& );

  protected:
    GateMergedVolumeActor(G4String name, G4int depth);

    GateMergedVolumeActorMessenger* pMergedVolumeActorMessenger;
    GateActorMessenger * pActorMessenger;

  private:
    std::vector<G4String>             mVolToMerge;
    std::vector<G4VSolid*>            mSolidVolToMerge;
    std::vector<G4VPhysicalVolume*>   mPhysicalVolToMerge;
    std::vector<G4LogicalVolume*>     mLogicalVolToMerge;
};

MAKE_AUTO_CREATOR_ACTOR(MergedVolumeActor,GateMergedVolumeActor)

#endif
