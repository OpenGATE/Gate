/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateFluenceActor :
  \brief
*/
// Gate
#include "GateFluenceActor.hh"
#include "GateScatterOrderTrackInformationActor.hh"

//-----------------------------------------------------------------------------
GateFluenceActor::GateFluenceActor(G4String name, G4int depth):
  GateVImageActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateFluenceActor() -- begin"<<G4endl);
  pMessenger = new GateFluenceActorMessenger(this);
  SetStepHitType("pre");
  mResponseFileName = "";
  GateDebugMessageDec("Actor",4,"GateFluenceActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateFluenceActor::~GateFluenceActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateFluenceActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct(); // mImage is not allocated here
  mImage.Allocate();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // the image index will be computed according to the preStep
  if (mStepHitType != PreStepHitType) {
    GateWarning("The stepHitType must be 'pre', we force it.");
    SetStepHitType("pre");
  }
  SetStepHitType("pre");

  // Read the response detector curve from an external file
  mEnergyResponse.ReadResponseDetectorFile(mResponseFileName);

  // Allocate scatter image
  if (mIsScatterImageEnabled) {
    mImageScatter.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageScatter.Allocate();
    mImageScatter.SetOrigin(mOrigin);
    mImageCompton.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageCompton.Allocate();
    mImageCompton.SetOrigin(mOrigin);
    mImageRayleigh.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageRayleigh.Allocate();
    mImageRayleigh.SetOrigin(mOrigin);
    mImageFluo.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageFluo.Allocate();
    mImageFluo.SetOrigin(mOrigin);
  }

  // Print information
  GateMessage("Actor", 1,
              "\tFluence FluenceActor    = '" << GetObjectName() << "'" << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateFluenceActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateFluenceActor::SaveData()
{
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  char filename[1024];
  // Printing all particles
  GateVImageActor::SaveData();
  if(mSaveFilename != "")
  {
    sprintf(filename, mSaveFilename, rID);
    mImage.Write(filename);
    // Printing just scatter
    if(mIsScatterImageEnabled)
    {
      G4String fn = removeExtension(filename)+"-scatter."+G4String(getExtension(filename));
      mImageScatter.Write(fn);
    }
  }
  // Printing scatter of each order
  if(mScatterOrderFilename != "")
  {
    for(unsigned int k = 0; k<mFluencePerOrderImages.size(); k++)
    {
      sprintf(filename, mScatterOrderFilename, rID, k+1);
      mFluencePerOrderImages[k]->Write((G4String)filename);
    }
  }

  // Printing scatter of each order
  if(mScatterProcessFilename != "")
  {
      sprintf(filename, "output/fluence_compton_%04d.mha", rID);
      mImageCompton.Write((G4String)filename);

      sprintf(filename, "output/fluence_rayleigh_%04d.mha", rID);
      mImageRayleigh.Write((G4String)filename);

      sprintf(filename, "output/fluence_fluo_%04d.mha", rID);
      mImageFluo.Write((G4String)filename);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::ResetData()
{
  mImage.Fill(0);
  if(mIsScatterImageEnabled) {
    mImageScatter.Fill(0);
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateFluenceActor::BeginOfRunAction( const G4Run *)
{
}
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
void GateFluenceActor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::UserSteppingActionInVoxel(const int index, const G4Step* step)
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel - begin" << G4endl);

  // Is this necessary?
  if(index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());

  /* http://geant4.org/geant4/support/faq.shtml
     To check that the particle has just entered in the current volume
     (i.e. it is at the first step in the volume; the preStepPoint is at the boundary):
  */
  if (step->GetPreStepPoint()->GetStepStatus() == fGeomBoundary)
    {
      double energy = (step->GetPreStepPoint()->GetKineticEnergy());
      double respValue = mEnergyResponse(energy);
           
      mImage.AddValue(index, respValue);

     if(mIsScatterImageEnabled) {
      unsigned int order = 0;
      G4String process = "";
      // Scatter order
      if(info) {
        order   = info->GetScatterOrder();
        process = info->GetScatterProcess();
        if(order) {
          while(order>mFluencePerOrderImages.size() && order>0) {
            GateImage * voidImage = new GateImage;
            voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
            voidImage->Allocate();
            voidImage->SetOrigin(mOrigin);
            voidImage->Fill(0);
            mFluencePerOrderImages.push_back( voidImage );
          }
        }
      }
      // Scattered primary particles, e.g., primary photons that undergo
      // Compton and Rayleigh interactions. Straight interactions are missed.
      if(!step->GetTrack()->GetParentID() &&
         !step->GetTrack()->GetDynamicParticle()->GetPrimaryParticle()
                                                ->GetMomentum().isNear(step->GetTrack()->GetDynamicParticle()->GetMomentum())) {
        mImageScatter.AddValue(index, respValue);

        if (process == G4String("Compton"))
          mImageCompton.AddValue(index, respValue);
        else if (process == G4String("RayleighScattering"))
          mImageRayleigh.AddValue(index, respValue);

        // Scatter order image
        if(order)
          mFluencePerOrderImages[order-1]->AddValue(index, respValue);
      }
      // Secondary particles, e.g., Fluorescence gammas
      if(step->GetTrack()->GetTrackID() && step->GetTrack()->GetParentID()>0 ) {
        mImageScatter.AddValue(index, respValue);
        // Scatter order image
        if (process == G4String("PhotoElectric"))
          mImageFluo.AddValue(index, respValue);
        if(order)
          mFluencePerOrderImages[order-1]->AddValue(index, respValue);
      }
    }
  }

  GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------
