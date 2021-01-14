/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateTrackLengthActor.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateTrackLengthActor::GateTrackLengthActor(G4String name, G4int depth) :
        GateVActor(name, depth) {
    GateDebugMessageInc("Actor", 4, "GateTrackLengthActor() -- begin\n");

    mLmin = 0.;
    mLmax = 50.;
    mNBins = 10;


    pMessenger = new GateTrackLengthActorMessenger(this);

    GateDebugMessageDec("Actor", 4, "GateTrackLengthActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateTrackLengthActor::~GateTrackLengthActor() {
    GateDebugMessageInc("Actor", 4, "~GateTrackLengthActor() -- begin\n");



    GateDebugMessageDec("Actor", 4, "~GateTrackLengthActor() -- end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateTrackLengthActor::Construct() {
    GateVActor::Construct();

    // Enable callbacks
    EnableBeginOfRunAction(true);
    EnableBeginOfEventAction(false);
    EnablePreUserTrackingAction(false);
    EnableUserSteppingAction(false);
    EnablePostUserTrackingAction(true);

    //mHistName = "Precise/output/TrackLength.root";
    pTfile = new TFile(mSaveFilename, "RECREATE");
    pTrackLength = new TH1D("trackLength", "TrackLength", GetNBins(), GetLmin(), GetLmax());
    pTrackLength->SetXTitle("Track Length (mm)");

    ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateTrackLengthActor::SaveData() {
    GateVActor::SaveData();
    pTfile->Write();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackLengthActor::ResetData() {
    pTrackLength->Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateTrackLengthActor::BeginOfRunAction(const G4Run *) {
    GateDebugMessage("Actor", 3, "GateTrackLengthActor -- Begin of Run\n");
    ResetData();
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateTrackLengthActor::PostUserTrackingAction(const GateVVolume * /*vol*/, const G4Track *aTrack) {
    pTrackLength->Fill(aTrack->GetTrackLength(), aTrack->GetWeight());

}
//-----------------------------------------------------------------------------

#endif
