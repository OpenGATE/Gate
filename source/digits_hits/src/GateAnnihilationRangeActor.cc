/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include <G4VProcess.hh>
#include "GateAnnihilationRangeActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateAnnihilationRangeActor::GateAnnihilationRangeActor(G4String name, G4int depth) :
        GateVActor(name, depth) {
    pMessenger = new GateAnnihilationRangeActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateAnnihilationRangeActor::~GateAnnihilationRangeActor() {
    delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateAnnihilationRangeActor::Construct() {
    GateVActor::Construct();

    // Enable callbacks
    EnablePostUserTrackingAction(true);

    mFile = new GateOutputTreeFileManager();
    mFile->add_file(mSaveFilename, "root"); // FIXME type
    mFile->set_tree_name("Annihilation");
    mFile->write_variable("X", &mX);
    mFile->write_variable("Y", &mY);
    mFile->write_variable("Z", &mZ);

    //  ResetData();
    mFile->write_header();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateAnnihilationRangeActor::SaveData() {
    GateVActor::SaveData();
    mFile->write();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateAnnihilationRangeActor::ResetData() {
    GateError("Can't reset GateAnnihilationRangeActor");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateAnnihilationRangeActor::PostUserTrackingAction(const GateVVolume * /*vol*/, const G4Track *aTrack) {

    // If this is not a e+ (id=319) we do nothing
    auto id = aTrack->GetParticleDefinition()->GetParticleName();
    if (id != "e+") return;

    /* debug
    std::cout << " post track part " << aTrack->GetParticleDefinition()->GetParticleName() << std::endl;
    std::cout << " post track part " << aTrack->GetParticleDefinition()->GetParticleDefinitionID() << std::endl;
    std::cout << " post track status " << aTrack->GetTrackStatus() << std::endl;
    auto creator = aTrack->GetCreatorProcess();
    if (creator)
        std::cout << " post track proc creator " << creator->GetProcessName() << std::endl;
    std::cout << " post track last step in vol " << aTrack->GetStep()->IsLastStepInVolume() << std::endl;
    auto step = aTrack->GetStep()->GetPostStepPoint();
    auto def_proc = step->GetProcessDefinedStep();
    if (def_proc)
        std::cout << " post track step proc " << def_proc->GetProcessName() << std::endl;
    */

    // last process should be annihil
    auto proc = aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep();
    if (proc->GetProcessName() != "annihil") return;

    // we store the 3D coordinates
    auto post_step = aTrack->GetStep()->GetPostStepPoint()->GetPosition();
    mX = post_step.getX();
    mY = post_step.getZ();
    mZ = post_step.getX();
    mFile->fill();

}
//-----------------------------------------------------------------------------
