/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaTLEActor.hh"
#include "GatePromptGammaTLEActorMessenger.hh"
#include "GateImageOfHistograms.hh"

#include <G4Proton.hh>
#include <G4VProcess.hh>

//-----------------------------------------------------------------------------
GatePromptGammaTLEActor::GatePromptGammaTLEActor(G4String name, G4int depth):
  GateVImageActor(name, depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaTLEActorMessenger(this);
  SetStepHitType("random");
  //mImageGamma = new GateImageOfHistograms("double");
  mCurrentEvent = -1;
  mIsUncertaintyImageEnabled = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaTLEActor::~GatePromptGammaTLEActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SetInputDataFilename(std::string filename)
{
  mInputDataFilename = filename;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::Construct()
{
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Input data
  data.Read(mInputDataFilename);
  data.InitializeMaterial();

  //set up and allocate lasthiteventimage
  SetOriginTransformAndFlagToImage(mLastHitEventImage);
  mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLastHitEventImage.Allocate();
  mLastHitEventImage.Fill(-1); //does allocate imply Filling with zeroes?

  //set up output images.
  std::vector<GateImageOfHistograms*> looplist;
  looplist.push_back(mImageGamma);
  if (mIsUncertaintyImageEnabled) looplist.push_back(tleuncertain);
  for (int i = 0; i < looplist.size(); i++) {
    looplist[i] = new GateImageOfHistograms("double");
    looplist[i]->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    looplist[i]->SetOrigin(mOrigin);
    looplist[i]->SetTransformMatrix(mImage.GetTransformMatrix());
    looplist[i]->SetHistoInfo(data.GetGammaNbBins(), data.GetGammaEMin(), data.GetGammaEMax());
    //list[i]->Allocate(); //lets save some memory!
  }

  //set up and allocate runtime images.
  looplist.clear();
  looplist.push_back(tmptrackl);
  looplist.push_back(trackl);
  looplist.push_back(tracklsq);
  for (int i = 0; i < looplist.size(); i++) {
    looplist[i] = new GateImageOfHistograms("double");
    looplist[i]->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    looplist[i]->SetOrigin(mOrigin);
    looplist[i]->SetTransformMatrix(mImage.GetTransformMatrix());
    looplist[i]->SetHistoInfo(data.GetProtonNbBins(), data.GetProtonEMin(), data.GetProtonEMax());
    looplist[i]->Allocate();
    looplist[i]->PrintInfo();
  }

  //set converterHist. sole use is to aid conversion of proton energy to bin index.
  converterHist = new TH1D("Ep", "proton energy", data.GetProtonNbBins(), data.GetProtonEMin() / MeV, data.GetProtonEMax() / MeV);

  // Force hit type to random
  if (mStepHitType != RandomStepHitType) {
    GateWarning("Actor '" << GetName() << "' : stepHitType forced to 'random'" << std::endl);
  }
  SetStepHitType("random");

  // Set to zero
  //ResetData(); //allocate implies reset
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::ResetData()
{
  //Does NOT reset mImageGamma, tleuncertain
  tmptrackl->Reset();
  trackl->Reset();
  tracklsq->Reset();
  mLastHitEventImage.Fill(-1);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SaveData()
{
  // Data are normalized by the number of primaries
  static bool alreadyHere = false;
  if (alreadyHere) {
    GateError("The GatePromptGammaTLEActor has already been saved and normalized. However, it must write its results only once. Remove all 'SaveEvery' for this actor. Abort.");
  }

  // Update (and allocate) mImageGamma, tleuncertain
  BuildOutput();

  // Normalisation, so that we have the numebr per proton, which is easier to use.
  int n = GateActorManager::GetInstance()->GetCurrentEventId() + 1; // +1 because start at zero
  double f = 1.0 / n;
  mImageGamma->Scale(f);
  GateVImageActor::SaveData();
  mImageGamma->Write(mSaveFilename);
  alreadyHere = true;

  if (mIsUncertaintyImageEnabled) {
  //TODO save tleuncertain
  }
  if (mIsIntermediaryUncertaintyOutputEnabled) {
  //TODO save trackl,tracklsq
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at start of each event
void GatePromptGammaTLEActor::BeginOfEventAction(const G4Event *e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GatePromptGammaTLEActor -- Begin of Event: " << mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPostTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPreTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserSteppingActionInVoxel(int index, const G4Step *step)
{
  // Check index
  if (index < 0) return;

  // compute sameEvent
  // sameEvent is false the first time some energy is deposited for each primary particle
  bool sameEvent=true;
  GateDebugMessage("Actor", 2,  "GateDoseActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
  if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
    sameEvent = false;
    mLastHitEventImage.SetValue(index, mCurrentEvent);
  }
  // Get information
  const G4ParticleDefinition *particle = step->GetTrack()->GetParticleDefinition();
  const G4double &particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double &distance = step->GetStepLength();

  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;

  // Check material
  //const G4Material *material = step->GetPreStepPoint()->GetMaterial();

  // Get value from histogram. We do not check the material index, and
  // assume everything exist (has been computed by InitializeMaterial)
  //TH1D *h = data.GetGammaEnergySpectrum(material->GetIndex(), particle_energy);

  // Check if proton energy within bounds.
  double dbmax = data.GetProtonEMax();
  if (particle_energy > dbmax) {
    GateError("GatePromptGammaTLEActor -- Proton Energy (" << particle_energy << ") outside range of pgTLE (" << dbmax << ") database! Aborting...");
  }

  // Do not scale h directly because it will be reused
  //mImageGamma->AddValueDouble(index, h, distance * material->GetDensity() / (g / cm3));

  int protbin = GetProtonBin(particle_energy);
  //if same event, then add to tmptrack
  if (sameEvent) tmptrackl->AddValueDouble(index, protbin, distance);
  //if not, then update trackl,tracklsq from the previous event, and restart tmptrackl.
  else {
    double tmp = tmptrackl->GetValueDouble(index, protbin);
    trackl->AddValueDouble(index, protbin, tmp);
    if (mIsUncertaintyImageEnabled) tracklsq->AddValueDouble(index, protbin, tmp*tmp);
    tmptrackl->SetValueDouble(index, protbin, distance);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Convert Proton Energy to a bin index.
int GatePromptGammaTLEActor::GetProtonBin(double energy) {
  return converterHist->Fill(energy / MeV);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::BuildOutput() {
  //allocate output images
  std::vector<GateImageOfHistograms*> looplist;
  looplist.push_back(mImageGamma);
  if (mIsUncertaintyImageEnabled) looplist.push_back(tleuncertain);
  for (int i = 0; i < looplist.size(); i++) {
    looplist[i]->Allocate();
    looplist[i]->PrintInfo();
  }

  //get pointers to images
  double *imImageGamma = mImageGamma->GetDataDoublePointer();
  double *itleuncertain = tleuncertain->GetDataDoublePointer();
  double *itmptrackl = tmptrackl->GetDataDoublePointer();
  double *itrackl = trackl->GetDataDoublePointer();
  double *itracklsq = tracklsq->GetDataDoublePointer();

  //update trackl,tracklsq. NOTE: this loop is over voxelindex,protonenergy
  for (unsigned long i = 0; i < tmptrackl->GetDoubleSize() ; i++) {
    //first, finalize trackl and tracklsq
    itrackl[i] += itmptrackl[i];
    itracklsq[i] += itmptrackl[i] * itmptrackl[i];
    itmptrackl[i] = 0.; //Reset
  }

  //compute TLE output. NOTE: this loop is over voxelindex,gammaenergy
  /*for(voxel){
      GateVImageVolume* phantom = GetPhantom();
    }

  // Check material
  //const G4Material *material = step->GetPreStepPoint()->GetMaterial();

  // Get value from histogram. We do not check the material index, and
  // assume everything exist (has been computed by InitializeMaterial)
  //TH1D *h = data.GetGammaEnergySpectrum(material->GetIndex(), particle_energy);

  //now, load mImageGamma and tleuncertain
  //mImageGamma->AddValueDouble(index, h, distance * material->GetDensity() / (g / cm3));
  for (unsigned long i = 0; i < mImageGamma->GetDoubleSize() ; i++) {
    imImageGamma = itrackl[i] * material->GetDensity() / (g / cm3) ;

    if (mIsUncertaintyImageEnabled) {
      itleuncertain = 0;
    }
  }*/
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVImageVolume* GatePromptGammaTLEActor::GetPhantom() {
  // Search for voxelized volume. If more than one, crash (yet).
  GateVImageVolume* gate_image_volume = NULL;
  for(std::map<G4String, GateVVolume*>::const_iterator it  = GateObjectStore::GetInstance()->begin();
                                                       it != GateObjectStore::GetInstance()->end();
                                                       it++)
    {
    if(dynamic_cast<GateVImageVolume*>(it->second))
    {
      if(gate_image_volume != NULL)
        GateError("There is more than one voxelized volume and don't know yet how to cope with this.");
      else
        gate_image_volume = dynamic_cast<GateVImageVolume*>(it->second);
    }

  }
  return gate_image_volume;
}
//-----------------------------------------------------------------------------
