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
#include "GateDetectorConstruction.hh"

#include <G4Proton.hh>
#include <G4VProcess.hh>

//-----------------------------------------------------------------------------
GatePromptGammaTLEActor::GatePromptGammaTLEActor(G4String name, G4int depth):
  GateVImageActor(name, depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaTLEActorMessenger(this);
  SetStepHitType("random");
  mCurrentEvent = -1;
  mIsVarianceImageEnabled = false;
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

  //set up and allocate runtime images.
  SetTLEIoH(mImageGamma);
  if (mIsVarianceImageEnabled){
    SetTrackIoH(tmptrackl);
    SetTrackIoH(trackl);
    SetTrackIoH(tracklsq);
  }

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
  //Does NOT reset mImageGamma, tleuncertain, it will allocate them.
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

  GateVImageActor::SaveData();  //What does this do?

  // Number of primaries for normalisation, so that we have the number per proton, which is easier to use.
  mImageGamma->Scale(1./(GateActorManager::GetInstance()->GetCurrentEventId() + 1));// +1 because start at zero
  mImageGamma->Write(mSaveFilename);

  if (mIsVarianceImageEnabled) {
    BuildOutput();
    tle->Write(G4String(removeExtension(mSaveFilename))+"-TLE."+G4String(getExtension(mSaveFilename)));
    tlevariance->Write(G4String(removeExtension(mSaveFilename))+"-TLEuncertainty."+G4String(getExtension(mSaveFilename)));
  }

  //optionally TODO output tracklengths
  alreadyHere = true;

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
  GateDebugMessage("Actor", 2,  "GatePromptGammaTLEActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
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

  // Check if proton energy within bounds.
  if (particle_energy > data.GetProtonEMax()) {
    GateError("GatePromptGammaTLEActor -- Proton Energy (" << particle_energy << ") outside range of pgTLE (" << data.GetProtonEMax() << ") database! Aborting...");
  }

  // New style TLE + TLE variance
  if (mIsVarianceImageEnabled) {
    int protbin = data.GetHEp()->FindFixBin(particle_energy)-1;
    if (!sameEvent) {
      //if not, then update trackl,tracklsq from the previous event, and restart tmptrackl.
      double tmp = tmptrackl->GetValueDouble(index, protbin);
      trackl->AddValueDouble(index, protbin, tmp);
      tracklsq->AddValueDouble(index, protbin, tmp*tmp);
      tmptrackl->SetValueDouble(index, protbin, distance);
    } else tmptrackl->AddValueDouble(index, protbin, distance);
  }

  // Old style TLE
  // Check material
  const G4Material *material = step->GetPreStepPoint()->GetMaterial();

  // Get value from histogram. We do not check the material index, and
  // assume everything exist (has been computed by InitializeMaterial)
  TH1D *h = data.GetGammaEnergySpectrum(material->GetIndex(), particle_energy);

  // Do not scale h directly because it will be reused
  mImageGamma->AddValueDouble(index, h, distance * material->GetDensity() / (g / cm3));
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::BuildOutput() {
  //nr primaries, +1 because start at zero, double because we will divide by it.
  double n = GateActorManager::GetInstance()->GetCurrentEventId() + 1;

  //allocate output images
  SetTLEIoH(tle);
  SetTLEIoH(tlevariance);

  //finalize trackl,tracklsq. NOTE: this loop is over voxelindex,protonenergy
  double *itmptrackl = tmptrackl->GetDataDoublePointer();
  double *itrackl = trackl->GetDataDoublePointer();
  double *itracklsq = tracklsq->GetDataDoublePointer();
  for (unsigned long i = 0; i < tmptrackl->GetDoubleSize() ; i++) {
    itrackl[i] += itmptrackl[i];
    itracklsq[i] += itmptrackl[i] * itmptrackl[i];
    itmptrackl[i] = 0.; //Reset
  }

  GateVImageVolume* phantom = GetPhantom(); //this has the correct label to material database.
  GateImage* phantomvox = phantom->GetImage(); //this has the array of voxels.
  //FIXME: pre-build label to G4Material map to save some time.

  //compute TLE output. first, loop over voxels
  for(unsigned int vi = 0; vi < tmptrackl->GetNumberOfValues() ;vi++ ){
    //PixelType label = phantomvox->GetValue(tmptrackl->GetCoordinatesFromIndex(vi));
    //convert between voxelsizes in phantom and output, NEAREST NEIGHBOUR!
    G4String materialname = phantom->GetMaterialNameFromLabel(phantomvox->GetValue(tmptrackl->GetVoxelCenterFromIndex(vi)));
    G4Material* material = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(materialname);
    int materialindex = material->GetIndex();

    TH2D* gammam = data.GetGammaM(materialindex);
    TH2D* ngammam = data.GetNgammaM(materialindex);

    // prep some things that are constant for all gamma bins
    double tracklav[data.GetProtonNbBins()];
    double tracklavsq[data.GetProtonNbBins()];
    double tracklsqsum[data.GetProtonNbBins()];
    double tracklvar[data.GetProtonNbBins()];
    for(int pi=0; pi<data.GetProtonNbBins() ; pi++ ){
      double trackli = trackl->GetValueDouble(vi,pi);
      if(trackli<=0.) { //if trackl==0, then all is zero.
        tracklav[pi] = 0.;
        tracklsqsum[pi] = 0.;
        tracklvar[pi] = 0.;
        tracklavsq[pi] = 0.;
        continue;
      }
      //if not, compute trackl,tracklsq,tracklavsq
      tracklav[pi] = trackli/n; //this is the sum(L)/n
      tracklsqsum[pi] = tracklsq->GetValueDouble(vi,pi); //this is the sum(L^2)
      tracklavsq[pi] = pow(tracklav[pi],2);

      //variance
      if (n==1.) tracklvar[pi]=trackli; //TODO check with JM.
      else tracklvar[pi] = tracklsqsum[pi]/n - pow(tracklav[pi],2);
      //else tracklvar[pi] = ( tracklsqsum[pi] - pow(trackli, 2)/n ) / (n-1.); //same as above (for large n, n=n-1)
    }

    for(int gi=0; gi<data.GetGammaNbBins() ; gi++ ){ //per proton bin, compute the contribution to the gammabin
      double tleval = 0.;
      double tleuncval = 0.;
      for(int pi=0; pi<data.GetProtonNbBins() ; pi++ ){
        if(tracklav[pi]==0.) {
          continue; //dont need to add anything to TLE or TLEunc.
        }
        double igammam = gammam->GetBinContent(pi+1,gi+1);
        double ingammam = ngammam->GetBinContent(pi+1,gi+1);
        //TLE, TLE uncertainty
        tleval += igammam * tracklav[pi];

        if (ingammam > 0.) tleuncval += ( pow(igammam,2) * ( tracklvar[pi]/n + tracklavsq[pi]/ingammam ) );
        else tleuncval += ( pow(igammam,2) * ( tracklvar[pi]/n + tracklavsq[pi] ) );
        //if (tleuncval!=tleuncval) tleuncval = 0.; //check for division by zero.
      }

      tle->SetValueDouble(vi,gi,tleval);
      tlevariance->SetValueDouble(vi,gi,tleuncval); //remember this is the SQUARE of stddev, must take root later.
      //we do this so we can sum variances and take sqrt and get proper stddev when integrating a dimension.
    }

    /* testloop, in case you want to use it:
     * disable tleuncertain->SetValueDouble(vi,gi,tleuncval)
     * enable setting dens(ity)
    for(int pi=0; pi<data.GetProtonNbBins() ; pi++ ){
      // assume everything exist (has been computed by InitializeMaterial)
      TH1D *h = data.GetGammaEnergySpectrum(materialindex, (double)pi/250.*200.+0.5*200./250.);
      tleuncertain->AddValueDouble(vi, h, tracklav[pi] * dens );
    }*/

  }//end voxelloop

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


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SetTLEIoH(GateImageOfHistograms*& ioh) {
  ioh = new GateImageOfHistograms("double");
  ioh->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  ioh->SetOrigin(mOrigin);
  ioh->SetTransformMatrix(mImage.GetTransformMatrix());
  ioh->SetHistoInfo(data.GetGammaNbBins(), data.GetGammaEMin(), data.GetGammaEMax());
  ioh->Allocate();
  ioh->PrintInfo();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SetTrackIoH(GateImageOfHistograms*& ioh) {
  ioh = new GateImageOfHistograms("double");
  ioh->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  ioh->SetOrigin(mOrigin);
  ioh->SetTransformMatrix(mImage.GetTransformMatrix());
  ioh->SetHistoInfo(data.GetProtonNbBins(), data.GetProtonEMin(), data.GetProtonEMax());
  ioh->Allocate();
  ioh->PrintInfo();
}
//-----------------------------------------------------------------------------
