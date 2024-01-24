/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  mIsDebugOutputEnabled = false;
  mIsOutputMatchEnabled = false;
  alreadyHere = false;
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
  data.InitializeMaterial(mIsDebugOutputEnabled);

  //set up and allocate runtime images.
  SetTLEIoH(mImageGamma);
  SetTofIoH(mImagetof);
  if (mIsDebugOutputEnabled){
    //set up and allocate lasthiteventimage
    SetOriginTransformAndFlagToImage(mLastHitEventImage);
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mLastHitEventImage.Fill(-1); //does allocate imply Filling with zeroes?

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
  if (alreadyHere) {
    GateError("The GatePromptGammaTLEActor has already been saved and normalized. However, it must write its results only once. Remove all 'SaveEvery' for this actor. Abort.");
  }

  //GateVImageActor::SaveData();  //What does this do?

  // Number of primaries for normalisation, so that we have the number per proton, which is easier to use.
  mImageGamma->Scale(1./(GateActorManager::GetInstance()->GetCurrentEventId() + 1));// +1 because start at zero
  mImageGamma->Write(mSaveFilename);

  //delete mImageGamma; 
  mImagetof->Scale(1./(GateActorManager::GetInstance()->GetCurrentEventId() + 1));// +1 because start at zero
  mImagetof->Write(G4String(removeExtension(mSaveFilename))+"-tof."+G4String(getExtension(mSaveFilename)));
  
  if (mIsDebugOutputEnabled) {
    BuildVarianceOutput();
    tle->Write(G4String(removeExtension(mSaveFilename))+"-debugtle."+G4String(getExtension(mSaveFilename)));
    tlevariance->Write(G4String(removeExtension(mSaveFilename))+"-debugvar."+G4String(getExtension(mSaveFilename)));
    trackl->Write(G4String(removeExtension(mSaveFilename))+"-debugtrackl."+G4String(getExtension(mSaveFilename)));
    tracklsq->Write(G4String(removeExtension(mSaveFilename))+"-debugtracklsq."+G4String(getExtension(mSaveFilename)));
  }

  alreadyHere = true;

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at start of each event
void GatePromptGammaTLEActor::BeginOfEventAction(const G4Event *e) {
  //std::cout << "Event Begin. Press any key to continue." << std::endl;
  //std::cin.get();
  GateVActor::BeginOfEventAction(e);
  mCurrentIndex = -1;
  mCurrentEvent++;
  startEvtTime = e->GetPrimaryVertex()->GetT0();
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
  // Check if we are inside the volume (YES THIS ACTUALLY NEEDS TO BE CHECKED).
  if (index<0) return;

  // Get information
  const G4ParticleDefinition *particle = step->GetTrack()->GetParticleDefinition();
  
  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;
  // if (step->GetTrack()->GetParentID() != 0) return; // Keep 2ndary protons

  const G4double &particle_energy_in = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double &particle_energy_out = step->GetPostStepPoint()->GetKineticEnergy();
  const G4double &distance = step->GetStepLength();
  G4double inputtof = step->GetPreStepPoint()->GetGlobalTime() - startEvtTime;
  G4double outputtof = step->GetPostStepPoint()->GetGlobalTime() - startEvtTime;
  //randomization
  G4double randomNumberTime = G4UniformRand();
  G4double randomNumberEnergy = G4UniformRand();
  G4double particle_energy = particle_energy_out + (particle_energy_in-particle_energy_out)*randomNumberEnergy;
  G4double tof = inputtof + (outputtof-inputtof)*randomNumberTime;

  // Check if proton energy within bounds.
  if (particle_energy > data.GetProtonEMax()) {
    GateError("GatePromptGammaTLEActor -- Proton Energy (" << particle_energy << ") outside range of pgTLE (" << data.GetProtonEMax() << ") database! Aborting...");
  }

  // Post computation TLE + TLE systematic + random variance (for the uncorrelated case, which is wrong).
  if (mIsDebugOutputEnabled) {
    // compute sameEvent
    // sameEvent is false the first time some energy is deposited for each primary particle
    bool sameEvent=true;
    GateDebugMessage("Actor", 2,  "GatePromptGammaTLEActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
    if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
      sameEvent = false;
      mLastHitEventImage.SetValue(index, mCurrentEvent);
    }
    int protbin = data.GetHEp()->FindFixBin(particle_energy)-1;
    if (!sameEvent) {
      //if not, then update trackl,tracklsq from the previous event, and restart tmptrackl.
      double tmp = tmptrackl->GetValueDouble(index, protbin);
      trackl->AddValueDouble(index, protbin, tmp);
      tracklsq->AddValueDouble(index, protbin, tmp*tmp);
      tmptrackl->SetValueDouble(index, protbin, distance);
    } else tmptrackl->AddValueDouble(index, protbin, distance);
  }

  // Regular TLE
  G4Material *material = step->GetPreStepPoint()->GetMaterial();

  /* Because step->GetPreStepPoint() and tmptrackl->GetVoxelCenterFromIndex(index) are different positions,
   * we may get different materials if the phantom-volume and tle-output-volume are different (size, offset, voxelsize).
   * GetPreStepPoint is more precise, so we keep that here, but in case it is necessary to have identical outputs,
   * uncomment the below 4 lines to get the material. */
  if(mIsOutputMatchEnabled) {
    GateVImageVolume* phantom = GetPhantom(); //this has the correct label to material database.
    GateImage* phantomvox = phantom->GetImage(); //this has the array of voxels.
    G4String materialname = phantom->GetMaterialNameFromLabel(phantomvox->GetValue(tmptrackl->GetVoxelCenterFromIndex(index)));
    material = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(materialname);
  }

  // Get value from histogram. We do not check the material index, and
  // assume everything exist (has been computed by InitializeMaterial)
  //particle_energy_rand = particle_energy_in + (particle_energy_out-particle_energy_in)*randomNumberEnergy;
  TH1D *h = data.GetGammaEnergySpectrum(material->GetIndex(), particle_energy); 

  if (h != NULL) { // NULL if material is "worldDefaultAir"
    double pg_stats[4];
    h->GetStats(pg_stats);
    double pg_sum = pg_stats[0];

    // // To print TH1D characteristics
    // G4cout << "GatePromptGammaTLEActor::UserSteppingActionInVoxel: lowEdge 1 = " << h->GetXaxis()->GetBinLowEdge(1)
    // 	   << " -- upEdge " << h->GetXaxis()->GetNbins() << " = " << h->GetXaxis()->GetBinUpEdge(h->GetXaxis()->GetNbins()) << G4endl;
    
    // Also take the particle weight into account
    double w = step->GetTrack()->GetWeight();

    // Do not scale h directly because it will be reused
    mImageGamma->AddValueDouble(index, h, w * distance * material->GetDensity() / (g / cm3));
    // (material is converted from internal units to g/cm3)

    //----------------------------------------------------------------------------------------------------------
    /** Modif Oreste **/
    pTime->Fill(tof);
  
    mImagetof->AddValueDouble(index, pTime, pg_sum * w * distance * material->GetDensity() / (g / cm3));
    // Record the input and output time in voxels and generate randomize time value between input and output time value /** Modif Oreste **/
    //if (index != mCurrentIndex) {
    //Here we record the time in the image of the previous voxel (mCurrentIndex) before to change the input time of the current voxel (index)
    //if (mCurrentIndex != -1) {
      //PreStepPoint of the current step after a change of index corresponds to the PostStepPoint of the last step in the previous index
      //outputtof = step->GetPreStepPoint()->GetGlobalTime() - startEvtTime;
      //tof = inputtof + (outputtof-inputtof)*randomNumberTime; //randomization
      //pTime->Fill(tof);
      //mImagetof->AddValueDouble(mCurrentIndex, pTime, w * distance * material->GetDensity() / (g / cm3));
    //}
    //Here we update the input time in voxel "index" which will be attributed to mCurrentIndex after "index" changing
    //inputtof = step->GetPreStepPoint()->GetGlobalTime() - startEvtTime;
    //mCurrentIndex = index;
    //}
    //Recording of the time for the last index (index = mCurrentIndex) of the event
    //if (inputtof == outputtof && step->GetPostStepPoint()->GetVelocity()==0){
    //outputtof = step->GetPostStepPoint()->GetGlobalTime() - startEvtTime;
    //tof = inputtof + (outputtof-inputtof)*randomNumberTime;
    //pTime->Fill(tof);
    //mImagetof->AddValueDouble(mCurrentIndex, pTime, w * distance * material->GetDensity() / (g / cm3));
    //}

    pTime->Reset();
  }
  //------------------------------------------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::BuildVarianceOutput() {
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
    //tmptrackl is set back to 0 if added to trackl,tracklsq, so any remaining nonzero value must be added
    itrackl[i] += itmptrackl[i];
    itracklsq[i] += itmptrackl[i] * itmptrackl[i];
    itmptrackl[i] = 0.; //Reset
  }

  GateVImageVolume* phantom = GetPhantom(); //this has the correct label to material database.
  GateImage* phantomvox = phantom->GetImage(); //this has the array of voxels.

  //compute TLE output. first, loop over voxels
  for(int vi = 0; vi < tmptrackl->GetNumberOfValues() ;vi++ ){
    //PixelType label = phantomvox->GetValue(tmptrackl->GetCoordinatesFromIndex(vi));
    //convert between voxelsizes in phantom and output, NEAREST NEIGHBOUR!
    G4String materialname = phantom->GetMaterialNameFromLabel(phantomvox->GetValue(tmptrackl->GetVoxelCenterFromIndex(vi)));
    G4Material* material = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(materialname);
    int materialindex = material->GetIndex();

    TH2D* gammam = data.GetGammaM(materialindex);
    TH2D* ngammam = data.GetNgammaM(materialindex);

    // prep some things that are constant for all gamma bins
    std::vector<double> tracklav(data.GetProtonNbBins());
    std::vector<double> tracklavsq(data.GetProtonNbBins());
    std::vector<double> tracklsqsum(data.GetProtonNbBins());
    std::vector<double> tracklvar(data.GetProtonNbBins());
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
      if (n==1.) tracklvar[pi]=trackli; //this is an overestimate, so results might be better.
      else tracklvar[pi] = tracklsqsum[pi]/n - pow(tracklav[pi],2);
      //else tracklvar[pi] = ( tracklsqsum[pi] - pow(trackli, 2)/n ) / (n-1.); //same as above (for large n, n=n-1)
    }

    for(int gi=0; gi<data.GetGammaNbBins() ; gi++ ){ //per proton bin, compute the contribution to the gammabin
      double tleval = 0.;
      double tlevarval = 0.;
      for(int pi=0; pi<data.GetProtonNbBins() ; pi++ ){
        if(tracklav[pi]==0.) {
          continue; //dont need to add anything to TLE or TLEunc.
        }
        double igammam = gammam->GetBinContent(pi+1,gi+1);
        double ingammam = ngammam->GetBinContent(pi+1,gi+1);
        //TLE, TLE uncertainty
        tleval += igammam * tracklav[pi];

        if (ingammam > 0.) tlevarval += ( pow(igammam,2) * ( tracklvar[pi]/n + tracklavsq[pi]/ingammam ) );
        else tlevarval += ( pow(igammam,2) * ( tracklvar[pi]/n + tracklavsq[pi] ) );
        //if (tleuncval!=tleuncval) tleuncval = 0.; //check for division by zero.
      }

      tle->SetValueDouble(vi,gi,tleval);
      tlevariance->SetValueDouble(vi,gi,tlevarval); //remember this is the SQUARE of stddev, must take root later.
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
/** Modif Oreste **/
void GatePromptGammaTLEActor::SetTofIoH(GateImageOfHistograms*& ioh) {
  pTime = new TH1D("","",data.GetTimeNbBins(),0,data.GetTimeTMax()); // fin bin set at 0*ns
  ioh = new GateImageOfHistograms("double");
  ioh->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  ioh->SetOrigin(mOrigin);
  ioh->SetTransformMatrix(mImage.GetTransformMatrix());
  //ioh->SetHistoInfo(data.GetTimeNbBins(), data.GetTimeTMax()/data.GetTimeNbBins(), data.GetTimeTMax());
  ioh->SetHistoInfo(data.GetTimeNbBins(), 0., data.GetTimeTMax()); // first bin = 0*ns assumed
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
