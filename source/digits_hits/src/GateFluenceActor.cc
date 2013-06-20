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
  if( !mResponseFileName) ReadResponseDetectorFile();

  // Allocate scatter image
  if (mIsScatterImageEnabled) {
    mImageScatter.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageScatter.Allocate();
    mImageScatter.SetOrigin(mOrigin);
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
void GateFluenceActor::BeginOfRunAction( const G4Run*r)
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
      double respValue = 1.;
      if( !mResponseFileName)
	{
	  double energy = (step->GetPreStepPoint()->GetKineticEnergy());

	  // Energy Response Detector (linear interpolation to obtain the right value from the list)
	  std::map< G4double, G4double >::iterator iterResponseMap = mUserResponseMap.end();
	  iterResponseMap =  mUserResponseMap.lower_bound( energy);
	  if(iterResponseMap == mUserResponseMap.begin() || iterResponseMap == mUserResponseMap.end())
	    {
	      G4cout << "Particle Energy outside the Response Detector list" << G4endl;
	      exit(1);
	    }
	  double upperEn = iterResponseMap->first;
	  double upperMu = iterResponseMap->second;
          iterResponseMap--;
	  double lowerEn = iterResponseMap->first;
	  double lowerMu = iterResponseMap->second;
	  // Interpolation result value corresponding to the incedent photon and to count into the voxel
	  respValue = ((( upperMu - lowerMu)/( upperEn - lowerEn)) * ( energy - upperEn) + upperMu);
	}

      mImage.AddValue(index, respValue);
     if(mIsScatterImageEnabled) {
      unsigned int order = 0;
      // Scatter order
      if(info) {
        order = info->GetScatterOrder();
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
      // Compton and Rayleigh Case (straight rayleighs are missed)
      if(!step->GetTrack()->GetParentID() &&
         !step->GetTrack()->GetDynamicParticle()->GetPrimaryParticle()
                                                ->GetMomentum().isNear(step->GetTrack()->GetDynamicParticle()->GetMomentum())) {
        mImageScatter.AddValue(index, 1);
        // Scatter order image
        if(order)
          mFluencePerOrderImages[order-1]->AddValue(index, 1);
      }
      // Fluorescence case
      if(step->GetTrack()->GetTrackID() && step->GetTrack()->GetParentID()>0 ) {
        mImageScatter.AddValue(index, 1);
        // Scatter order image
        if(order)
          mFluencePerOrderImages[order-1]->AddValue(index, 1);
      }
    }
  }

  GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateFluenceActor::ReadResponseDetectorFile()
{
  G4double energy, response;
  std::ifstream inResponseFile;
  mUserResponseMap.clear( );

  inResponseFile.open( mResponseFileName);
  if( !inResponseFile )
    {
      // file couldn't be opened
      G4cout << "Error: file could not be opened" << G4endl;
      exit( 1);
    }
  while ( !inResponseFile.eof( ))
    {
      inResponseFile >> energy >> response;
      energy = energy*MeV;
      mUserResponseMap[ energy] = response;
    }
  inResponseFile.close( );
}
//-----------------------------------------------------------------------------

