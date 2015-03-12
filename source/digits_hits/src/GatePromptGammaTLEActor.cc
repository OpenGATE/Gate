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

  //set up and allocate runtime images.
  SetTLEIoH(mImageGamma);
  if (mIsUncertaintyImageEnabled){
    SetTrackIoH(tmptrackl);
    SetTrackIoH(trackl);
    SetTrackIoH(tracklsq);
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

  BuildOutput();
  GateVImageActor::SaveData();  //What does this do?
  if (mIsUncertaintyImageEnabled) {
    tle->Write(G4String(removeExtension(mSaveFilename))+"-TLE."+G4String(getExtension(mSaveFilename)));
    tleuncertain->Write(G4String(removeExtension(mSaveFilename))+"-TLEuncertainty."+G4String(getExtension(mSaveFilename)));
  }
  mImageGamma->Write(mSaveFilename);
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

  // Check if proton energy within bounds.
  if (particle_energy > data.GetProtonEMax()) {
    GateError("GatePromptGammaTLEActor -- Proton Energy (" << particle_energy << ") outside range of pgTLE (" << data.GetProtonEMax() << ") database! Aborting...");
  }

  if (mIsUncertaintyImageEnabled) {
    int protbin = GetProtonBin(particle_energy);
    if (sameEvent) tmptrackl->AddValueDouble(index, protbin, distance);
    //if not, then update trackl,tracklsq from the previous event, and restart tmptrackl.
    else {
      double tmp = tmptrackl->GetValueDouble(index, protbin);
      trackl->AddValueDouble(index, protbin, tmp);
      tracklsq->AddValueDouble(index, protbin, tmp*tmp);
      tmptrackl->SetValueDouble(index, protbin, distance);
    }
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
// Convert Proton Energy to a bin index.
int GatePromptGammaTLEActor::GetProtonBin(double energy) {
  return converterHist->Fill(energy / MeV);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::BuildOutput() {
  // Number of primaries for normalisation, so that we have the number per proton, which is easier to use.
  int n = GateActorManager::GetInstance()->GetCurrentEventId() + 1; // +1 because start at zero

  if (mIsUncertaintyImageEnabled) {
    //allocate output images
    SetTLEIoH(tle);
    SetTLEIoH(tleuncertain);

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
      //PixelType label = phantomvox->GetValue(tmptrackl->GetCoordinatesFromIndex(vi)); //convert between voxelsizes in phantom and output
      G4String materialname = phantom->GetMaterialNameFromLabel(phantomvox->GetValue(tmptrackl->GetCoordinatesFromIndex(vi)));
      G4Material* material = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(materialname);
      int materialindex = material->GetIndex();

      for(int gi=0; gi<data.GetGammaNbBins() ; gi++ ){ //per proton bin, compute the contribution to the gammabin
        TH1D* protonhist = data.GetGammaMForGammaBin(materialindex,gi);
        TH1D* ngammahist = data.GetNgammaMForGammaBin(materialindex,gi);
        double tleval = 0;
        double tleuncval = 0;
        for(int pi=0; pi<data.GetProtonNbBins() ; pi++ ){ //loop over gammabins to fill them up.

          //TLE output
          double tracklength = trackl->GetValueDouble(vi,pi); //this is the sum.
          double tracklengthsq = tracklsq->GetValueDouble(vi,pi); //this is the sum.
          double gammam = protonhist->GetBinContent(gi+1);
          tleval += gammam*tracklength; //+1 for TH1 offset

          //TLE uncertainty output
          //*po = sqrt( (1.0/(n-1))*(squared/n - pow(mean/n, 2)))/(mean/n); //from Dose Unc.
          double tlevar = sqrt( (1.0/(n-1))*(tracklengthsq/n - pow(tracklength/n, 2)))/(tracklength/n);
          double tleav = pow(tracklength,2)/n;
          double ngamma = ngammahist->GetBinContent(gi+1);
          tleuncval += pow(gammam,2) *( tlevar/n + pow(tleav,2)/ngamma );

        }

        tle->SetValueDouble(vi,gi,tleval); //we've now computed the sum, scale at the end with n.
        tleuncertain->SetValueDouble(vi,gi,tleuncval); //this has already been scaled.
      }
    }
    tle->Scale(1./n);
  }//endif uncertainty
  mImageGamma->Scale(1./n);

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
