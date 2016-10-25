/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateDoseSourceActor.hh"
#include "GateDoseSourceActorMessenger.hh"
#include "GateImageOfHistograms.hh"

//-----------------------------------------------------------------------------
GateDoseSourceActor::GateDoseSourceActor(G4String name, G4int depth):
  GateVImageActor(name, depth)
{
  pMessenger = new GateDoseSourceActorMessenger(this);
  //SetStepHitType("random");
  doseSourceImage = new GateImageOfHistograms("double"); //TODO: maybe float?
  bID = -1;
  bSourceName = " ";
  
  areweinityet = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDoseSourceActor::~GateDoseSourceActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActor::Construct()
{
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Force hit type to random
  if (mStepHitType != RandomStepHitType) {
    GateWarning("Actor '" << GetName() << "' : stepHitType forced to 'random'" << std::endl);
    SetStepHitType("random");
  }

  // Set to zero
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActor::ResetData()
{
  doseSourceImage->Reset();
}
//-----------------------------------------------------------------------------


// --------------------------------------------------------------------
void GateDoseSourceActor::BeginOfEventAction(const G4Event *) {
  //----------------------- Init beam and image ------------------------
  // because sources init after actors, we cannot request nbspots and allocate doseSpotIDimage any sooner than here
  if(areweinityet==false){
    doseSourceImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    doseSourceImage->SetOrigin(mOrigin);
    doseSourceImage->SetTransformMatrix(mImage.GetTransformMatrix());

    //need to get nrspots from source here
    tpspencilsource = dynamic_cast<GateSourceTPSPencilBeam *>(GateSourceMgr::GetInstance()->GetSourceByName( bSourceName ));
    if(SpotOrNot){
      int nbspots = tpspencilsource->GetTotalNumberOfSpots();
      doseSourceImage->SetHistoInfo(nbspots, 0, nbspots-1);
      //DD("brent, nbspots" << nbspots)
      doseSourceImage->Allocate();
      doseSourceImage->PrintInfo();
    } else {
      int nblayers = tpspencilsource->GetTotalNumberOfLayers();
      doseSourceImage->SetHistoInfo(nblayers, 0, nblayers-1);
      //DD("brent, nblayers" << nblayers)
      doseSourceImage->Allocate();
      doseSourceImage->PrintInfo();
        
    }
    areweinityet=true;
  }
  //-------------------------------------------------------------------
  
  //----------------------- Set SourceID ------------------------
  if(SpotOrNot){
    bID = tpspencilsource->GetCurrentSpotID();
  } else {
    bID = tpspencilsource->GetCurrentLayerID();
  }
  //-------------------------------------------------------------------
}
// --------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActor::SaveData()
{
  //GateVImageActor::SaveData();
  doseSourceImage->Write(mSaveFilename);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActor::UserPostTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSourceActor::UserPreTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActor::UserSteppingActionInVoxel(int index, const G4Step *step)
{
  if (index < 0) { // Check if we are inside the volume (YES THIS ACTUALLY NEEDS TO BE CHECKED).
    GateDebugMessage("Actor", 5, "index < 0 : do nothing\n");
    GateDebugMessageDec("Actor", 4, "GateDoseSourceActor -- UserSteppingActionInVoxel -- end\n");
    return;
  }
  
  const double weight = step->GetTrack()->GetWeight();
  const double edep = step->GetTotalEnergyDeposit()*weight;//*step->GetTrack()->GetWeight();
  
  if (edep == 0) { // if no energy is deposited => do nothing
    GateDebugMessage("Actor", 5, "edep == 0 : do nothing\n");
    GateDebugMessageDec("Actor", 4, "GateDoseSourceActor -- UserSteppingActionInVoxel -- end\n");
    return;
  }

  //---------------------------------------------------------------------------------
  // Volume weighting
  double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
  //---------------------------------------------------------------------------------

  //---------------------------------------------------------------------------------
  // Mass weighting
  //if(mDoseAlgorithmType == "MassWeighting")
  //  density = mVoxelizedMass.GetVoxelMass(index)/mDoseImage.GetVoxelVolume();
  //---------------------------------------------------------------------------------

  double dose = edep/density/doseSourceImage->GetVoxelVolume()/gray;
  
  doseSourceImage->AddValueDouble(index, bID, dose);
  
}
//-----------------------------------------------------------------------------

