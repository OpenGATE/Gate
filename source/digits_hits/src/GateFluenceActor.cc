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

  mCurrentEvent=-1;
  mIsSquaredImageEnabled = false;
  mIsUncertaintyImageEnabled = false;
  mIsLastHitEventImageEnabled = false;
  mIsNormalisationEnabled = false;
  mIsNumberOfHitsImageEnabled = false;

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
  
  mImageFilename = G4String(removeExtension(mSaveFilename))+"."+G4String(getExtension(mSaveFilename));
  mImageScatterFilename = G4String(removeExtension(mSaveFilename))+"-scatter."+G4String(getExtension(mSaveFilename));
  mNbOfHitsFilename = G4String(removeExtension(mSaveFilename))+"-NbOfHits."+G4String(getExtension(mSaveFilename));

  mImage.SetOrigin(mOrigin);
  mImageScatter.SetOrigin(mOrigin);
  mLastHitEventImage.SetOrigin(mOrigin);
  mNumberOfHitsImage.SetOrigin(mOrigin);

  mImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
  mImageScatter.SetOverWriteFilesFlag(mOverWriteFilesFlag);

  if( mIsSquaredImageEnabled || mIsUncertaintyImageEnabled)
    {
      mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mLastHitEventImage.Allocate();
      mIsLastHitEventImageEnabled = true;
    }

  mImage.EnableSquaredImage(mIsSquaredImageEnabled);
  mImage.EnableUncertaintyImage(mIsUncertaintyImageEnabled);
  // Force the computation of squared image if uncertainty is enabled
  if (mIsUncertaintyImageEnabled) mImage.EnableSquaredImage(true);
  mImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mImage.Allocate();
  mImage.SetFilename(mImageFilename);
  
  // Allocate scatter image
  if( mIsScatterImageEnabled)
    {
      mImageScatter.EnableSquaredImage(mIsSquaredImageEnabled);
      mImageScatter.EnableUncertaintyImage(mIsUncertaintyImageEnabled);
      // Force the computation of squared image if uncertainty is enabled
      if (mIsUncertaintyImageEnabled) mImageScatter.EnableSquaredImage(true);
      mImageScatter.SetResolutionAndHalfSize( mResolution, mHalfSize, mPosition);
      mImageScatter.Allocate();
      mImageScatter.SetFilename(mImageScatterFilename);
    }

  if (mIsNumberOfHitsImageEnabled)
    {
      mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mNumberOfHitsImage.Allocate();
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

      if( mIsNormalisationEnabled) mImage.SaveData(mCurrentEvent+1, true);
      else mImage.SaveData(mCurrentEvent+1, false);

      if( mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.Write(mNbOfHitsFilename);

      // Printing just scatter
      if( mIsScatterImageEnabled)
	{
          if( mIsNormalisationEnabled) mImageScatter.SaveData(mCurrentEvent+1, true);
          else mImageScatter.SaveData(mCurrentEvent+1, false);

          if( mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
	}
      if( mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1); // reset
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
  
  // Printing compton or rayleigh or fluorescence scatter images
  if(mSeparateScatteringFilename != "")
    {
      std::map<G4String,GateImage*>::iterator it = mInteractions.end();
      std::vector<G4String> interactions;
      interactions.push_back(G4String("Compton"));
      interactions.push_back(G4String("RayleighScattering"));
      interactions.push_back(G4String("PhotoElectric"));
      std::vector<G4String> interactionName;
      interactionName.push_back(G4String("_Compton.mhd"));
      interactionName.push_back(G4String("_Rayleigh.mhd"));
      interactionName.push_back(G4String("_Fluorescence.mhd"));
      
      // Saving separately scattering images (e.g. Compton, Rayleigh...)
      for(unsigned int i = 0; i<interactions.size(); i++){
	it = mInteractions.find(interactions[i]);
	if(it!=mInteractions.end()){
	  stringstream filenamestream;
	  filenamestream << mSeparateScatteringFilename << interactionName[i];
	  sprintf(filename, filenamestream.str().c_str(), rID);
	  mInteractions[interactions[i]]->Write((G4String)filename);
	}
	it = mInteractions.end();
      }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::ResetData()
{
  if( mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  mImage.Reset();
  mImageScatter.Reset();
  mImage.Fill(0);
  if(mIsScatterImageEnabled) mImageScatter.Fill(0);
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateFluenceActor::BeginOfRunAction( const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateFluenceActor -- Begin of Run" << G4endl);
}
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
void GateFluenceActor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateFluenceActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::UserSteppingActionInVoxel(const int index, const G4Step* step)
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel - begin" << G4endl);
  const double weight = step->GetTrack()->GetWeight();
  // Is this necessary?
  if(index <0)
    {
      GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
      GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
      return;
    }
  
  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());
  
  /* http://geant4.org/geant4/support/faq.shtml
     To check that the particle has just entered in the current volume
     (i.e. it is at the first step in the volume; the preStepPoint is at the boundary):
  */
  if( step->GetPreStepPoint()->GetStepStatus() == fGeomBoundary)
    {
      double energy = (step->GetPreStepPoint()->GetKineticEnergy());
      double respValue = mEnergyResponse(energy);
      bool sameEvent=true;
      if( mIsLastHitEventImageEnabled)
	{
	  GateDebugMessage( "Actor", 2,  "GateFluenceActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
	  if( mCurrentEvent != mLastHitEventImage.GetValue( index))
	    {
	      sameEvent = false;
	      mLastHitEventImage.SetValue(index, mCurrentEvent);
	    }
	}
      
      if( mIsUncertaintyImageEnabled || mIsSquaredImageEnabled)
	{
          if( sameEvent) mImage.AddTempValue( index, respValue);
	  else mImage.AddValueAndUpdate( index, respValue);
	}
      else mImage.AddValue( index, respValue);

      if( mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.AddValue( index, weight);

      
      if( mIsScatterImageEnabled) {
	unsigned int order = 0;
	G4String process = "";
	// Scatter order
	if(info) {
	  order   = info->GetScatterOrder();
	  process = info->GetScatterProcess();
	  // Allocate GateImage if process occurs
          if( process == G4String("Compton")){
	    GateImage * voidImage = new GateImage;
	    voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	    voidImage->Allocate();
	    voidImage->SetOrigin(mOrigin);
	    voidImage->Fill(0);
	    mInteractions.insert(std::pair<G4String,GateImage*>(process, voidImage));
	  }
          else if( process == G4String("RayleighScattering")){
	    GateImage * voidImage = new GateImage;
	    voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	    voidImage->Allocate();
	    voidImage->SetOrigin(mOrigin);
	    voidImage->Fill(0);
	    mInteractions.insert(std::pair<G4String,GateImage*>(process, voidImage));
	  }
          else if( process == G4String("PhotoElectric")){
	    GateImage * voidImage = new GateImage;
	    voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
	    voidImage->Allocate();
	    voidImage->SetOrigin(mOrigin);
	    voidImage->Fill(0);
	    mInteractions.insert(std::pair<G4String,GateImage*>(process, voidImage));
	  }
	  if(order) {
	    while(order > mFluencePerOrderImages.size() && order > 0) {
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
           ->GetMomentum().isNear(step->GetTrack()->GetDynamicParticle()->GetMomentum()))
          {

            if( mIsUncertaintyImageEnabled || mIsSquaredImageEnabled)
              {
                if( sameEvent) mImageScatter.AddTempValue( index, respValue);
                else mImageScatter.AddValueAndUpdate( index, respValue);
              }
            else mImageScatter.AddValue( index, respValue);

            if( process == G4String("Compton") || process == G4String("RayleighScattering") )
              mInteractions[process]->AddValue(index, respValue);
	  
            // Scatter order image
            if(order)
              mFluencePerOrderImages[order-1]->AddValue(index, respValue);
          }
	// Secondary particles, e.g., Fluorescence gammas
        if(step->GetTrack()->GetTrackID() && step->GetTrack()->GetParentID()>0 )
          {
            if( mIsUncertaintyImageEnabled || mIsSquaredImageEnabled)
              {
                if( sameEvent) mImageScatter.AddTempValue( index, respValue);
                else mImageScatter.AddValueAndUpdate( index, respValue);
              }
            else mImageScatter.AddValue( index, respValue);

            // Scatter order image
            if( process == G4String("PhotoElectric") )
              mInteractions[process]->AddValue(index, respValue);
            if(order)
              mFluencePerOrderImages[order-1]->AddValue(index, respValue);
          }
        }
      }
  GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------
