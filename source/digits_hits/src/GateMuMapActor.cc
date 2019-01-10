/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

//#include "GateConfiguration.h"

/*
   \class GateMuMapActor
   \author gsizeg@gmail.com
   \brief Class GateMuMapActor : This actor produces voxelised images of the heat diffusion in tissue.

   Parameters of the simulation given by the User in the macro:
   - setEnergy: Set energy
   */

#include "GateMuMapActor.hh"
#include "GateMuMapActorMessenger.hh"
#include "GateMaterialMuHandler.hh"

#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4UnitsTable.hh"
//-----------------------------------------------------------------------------

GateMuMapActor::GateMuMapActor(G4String name, G4int depth):
    GateVImageActor(name,depth) {
        GateDebugMessageInc("Actor",4,"GateMuMapActor() -- begin"<<G4endl);

        mEnergy = 0.511*MeV;
        mMuUnit= 1.0*(1.0/cm);
        pMessenger = new GateMuMapActorMessenger(this);
        GateDebugMessageDec("Actor",4,"GateMuMapActor() -- end"<<G4endl);
    }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor
GateMuMapActor::~GateMuMapActor()  {
    delete pMessenger;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

void GateMuMapActor::SetEnergy(G4double energy)
{
    mEnergy=energy;
}
//-----------------------------------------------------------------------------

void GateMuMapActor::SetMuUnit(G4double unit)
{
    mMuUnit=unit;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/// Constructor
void GateMuMapActor::Construct() {

    GateDebugMessageInc("Actor", 4, "GateMuMapActor -- Construct - begin" << G4endl);

    GateVImageActor::Construct();

    // Enable callbacks
    EnableBeginOfRunAction(true);
    EnableEndOfRunAction(true); // for save
    EnableBeginOfEventAction(true);
    EnableEndOfEventAction(false);
    EnablePreUserTrackingAction(false);
    EnableUserSteppingAction(false);

    // Output Filenames
    mMuMapFilename = G4String(removeExtension(mSaveFilename))+"-MuMap."+G4String(getExtension(mSaveFilename));
    mSourceMapFilename = G4String(removeExtension(mSaveFilename))+"-SourceMap."+G4String(getExtension(mSaveFilename));

    // Set origin, transform, flag
    SetOriginTransformAndFlagToImage(mMuMapImage);
    SetOriginTransformAndFlagToImage(mSourceMapImage);

    // Resize and allocate images
    mMuMapImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mMuMapImage.Allocate();

    mSourceMapImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mSourceMapImage.Allocate();

    // Print information
    GateMessage("Actor", 1,
            "\tMuMapActor    = '" << GetObjectName() << "'" << G4endl <<
            "\tMuMapFilename      = " << mMuMapFilename<< G4endl);

    ResetData();
    GateMessageDec("Actor", 4, "GateMuMapActor -- Construct - end" << G4endl);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateMuMapActor::SaveData() {

    mMuMapImage.Write(mMuMapFilename);
    mSourceMapImage.Write(mSourceMapFilename);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMuMapActor::ResetData() {

    mMuMapImage.Fill(0.0f);
    mSourceMapImage.Fill(0);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMuMapActor::BeginOfRunAction(const G4Run * r) {
    GateVActor::BeginOfRunAction(r);

    GateDebugMessage("Actor", 3, "GateMuMapActor -- Begin of Run" << G4endl);

    G4Navigator* theNavigator =G4TransportationManager::GetTransportationManager()
        ->GetNavigatorForTracking();
    for(int index=0; index<mMuMapImage.GetNumberOfValues(); index++)
    {
        G4ThreeVector myPosition=mMuMapImage.GetVoxelCenterFromIndex(index);
        G4VPhysicalVolume* pVolume = theNavigator->LocateGlobalPointAndSetup(myPosition);
        if(pVolume)
        {
        G4LogicalVolume* lVolume= pVolume->GetLogicalVolume();

        // Get the Mu value ,default unit(cm-1)
        double mu= GateMaterialMuHandler::GetInstance()->GetMu(lVolume->GetMaterialCutsCouple(),mEnergy)*(1/cm)/mMuUnit;

        mMuMapImage.SetValue(index,(float)mu);
        }
    }
    // Save MuMap voxel 
    SaveData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMuMapActor::EndOfRunAction(const G4Run* r)
{
    GateVActor::EndOfRunAction(r);
    SaveData();
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMuMapActor::BeginOfEventAction(const G4Event * e) {
    GateVActor::BeginOfEventAction(e);

    GateDebugMessage("Actor", 3, "GateMuMapActor -- Begin of Event: "<<mCurrentEvent << G4endl);
   G4ThreeVector primaryPosition = e->GetPrimaryVertex()->GetPosition();
   int index = mSourceMapImage.GetIndexFromPosition(primaryPosition);
   if(index>=0)
   {
       int count = mSourceMapImage.GetValue(index);
        mSourceMapImage.SetValue(index,count+1);
   }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMuMapActor::UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*track*/) {
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMuMapActor::UserSteppingActionInVoxel(const int /*index*/, const G4Step* /*step*/) {

}
//-----------------------------------------------------------------------------


