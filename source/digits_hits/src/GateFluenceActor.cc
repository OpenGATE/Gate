/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \brief Class GateFluenceActor :
  \brief
*/
/* Gate */
#include "GateFluenceActor.hh"
#include "GateScatterOrderTrackInformationActor.hh"
#include <sstream>
GateFluenceActor::GateFluenceActor(G4String name, G4int depth) :
  GateVImageActor(name, depth)
{
  GateDebugMessageInc("Actor",4,"GateFluenceActor() -- begin\n");
  mCurrentEvent = -1;
  mIsSquaredImageEnabled = false;
  mIsUncertaintyImageEnabled = false;
  mIsLastHitEventImageEnabled = false;
  mIsNormalisationEnabled = false;
  mIsNumberOfHitsImageEnabled = false;
  pMessenger = new GateFluenceActorMessenger(this);
  SetStepHitType("pre");
  mResponseFileName = "";
  mIgnoreWeight = false;
  GateDebugMessageDec("Actor",4,"GateFluenceActor() -- end\n");
}

/* Destructor */
GateFluenceActor::~GateFluenceActor()
{
  delete pMessenger;
}

/* Construct */
void GateFluenceActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- Construct - begin\n");
  GateVImageActor::Construct(); /* mImage is not allocated here */
  mImage.Allocate();

  /* Enable callbacks */
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);

  EnableUserSteppingAction(true);

  /* the image index will be computed according to the preStep */
  if (mStepHitType != PreStepHitType)
    {
      GateWarning("The stepHitType must be 'pre', we force it.");
      SetStepHitType("pre");
    }
  SetStepHitType("pre");

  /* Read the response detector curve from an external file */
  mEnergyResponse.ReadResponseDetectorFile(mResponseFileName);

  mImage.SetOrigin(mOrigin);
  mImageProcess.SetOrigin(mOrigin);
  mLastHitEventImage.SetOrigin(mOrigin);
  mNumberOfHitsImage.SetOrigin(mOrigin);

  mImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
  mImageProcess.SetOverWriteFilesFlag(mOverWriteFilesFlag);

  if (mIsSquaredImageEnabled || mIsUncertaintyImageEnabled)
    {
      mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mLastHitEventImage.Allocate();
      mIsLastHitEventImageEnabled = true;
    }

  mImage.EnableSquaredImage(mIsSquaredImageEnabled);
  mImage.EnableUncertaintyImage(mIsUncertaintyImageEnabled);
  /* Force the computation of squared image if uncertainty is enabled */
  if (mIsUncertaintyImageEnabled)
    {
      mImage.EnableSquaredImage(true);
    }
  mImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mImage.Allocate();

  /* Allocate scatter image */
  if (mIsScatterImageEnabled)
    {
      SetOriginTransformAndFlagToImage(mImageProcess);
      mImageProcess.EnableSquaredImage(mIsSquaredImageEnabled);
      mImageProcess.EnableUncertaintyImage(mIsUncertaintyImageEnabled);
      /* Force the computation of squared image if uncertainty is enabled */
      if (mIsUncertaintyImageEnabled)
        {
          mImageProcess.EnableSquaredImage(true);
        }
      mImageProcess.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mImageProcess.Allocate();
      mImageProcess.SetOrigin(mOrigin);
    }

  if (mIsNumberOfHitsImageEnabled)
    {
      SetOriginTransformAndFlagToImage(mNumberOfHitsImage);
      mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mNumberOfHitsImage.Allocate();
    }

  /* Print information */
  GateMessage("Actor", 1, "\tFluence FluenceActor    = '" << GetObjectName() << "'\n");

  ResetData();
  GateMessageDec("Actor", 4, "GateFluenceActor -- Construct - end\n");
}

/* Save data */
void GateFluenceActor::SaveData()
{
  G4int rID = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  char filename[1024];
  /* Printing all particles */
  GateVImageActor::SaveData();

  if (mSaveFilename != "")
    {
      sprintf(filename, mSaveFilename, rID);
      mImage.SetFilename(G4String(filename));
      if (mIsNormalisationEnabled)
        {
          mImage.SaveData(mCurrentEvent + 1, true);
        }
      else
        {
          mImage.SaveData(mCurrentEvent + 1, false);
        }

      if (mIsNumberOfHitsImageEnabled)
        {
          G4String fn = G4String(removeExtension(mSaveFilename))
            + "-NbOfHits."
            + G4String(getExtension(mSaveFilename));
          mNumberOfHitsImage.Write(fn);
        }

      /* Printing just scatter */
      if (mIsScatterImageEnabled)
        {
          G4String fn = removeExtension(filename) + "-scatter." + G4String(getExtension(filename));
          mImageProcess.SetFilename(fn);
          if (mIsNormalisationEnabled)
            {
              mImageProcess.SaveData(mCurrentEvent + 1, true);
            }
          else
            {
              mImageProcess.SaveData(mCurrentEvent + 1, false);
            }

          if (mIsLastHitEventImageEnabled)
            {
              mLastHitEventImage.Fill(-1);
            }
        }
      if (mIsLastHitEventImageEnabled)
        {
          mLastHitEventImage.Fill(-1);
        }/* reset */
    }

  /* Printing scatter of each order */
  if (mScatterOrderFilename != "")
    {
      for (unsigned int k = 0; k < mFluencePerOrderImages.size(); k++)
        {
          sprintf(filename, mScatterOrderFilename, rID, k + 1);
          mFluencePerOrderImages[k]->Write((G4String) filename);
        }
    }

  /* Printing compton or rayleigh or fluorescence scatter images */
  if (mSeparateProcessFilename != "")
    {
      /* Saving separately process images (e.g. Compton, Rayleigh...) */
      std::map<G4String, GateImage*>::iterator it = mProcesses.end();
      for (unsigned int i = 0; i < mProcessName.size(); i++)
        {
          it = mProcesses.find(mProcessName[i]);
          if (it != mProcesses.end())
            {
              std::stringstream filenamestream;
              filenamestream << mSeparateProcessFilename << "_" << mProcessName[i] << ".mhd";
              sprintf(filename, filenamestream.str().c_str(), rID);
              mProcesses[mProcessName[i]]->Write((G4String) filename);
            }
          it = mProcesses.end();
        }
    }
}

void GateFluenceActor::ResetData()
{
  if (mIsLastHitEventImageEnabled)
    {
      mLastHitEventImage.Fill(-1);
    }
  mImage.Reset();
  mImageProcess.Reset();
  mImage.Fill(0);
  if (mIsScatterImageEnabled)
    {
      mImageProcess.Fill(0);
    }
}

void GateFluenceActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateFluenceActor -- Begin of Run\n");
}

void GateFluenceActor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateFluenceActor -- Begin of Event: "<<mCurrentEvent << Gateendl);
}

void GateFluenceActor::UserPostTrackActionInVoxel(const int /*index*/,
                                                  const G4Track * /*aTrack*/)
{

  // Nothing (but must be implemented because virtual)
}

void GateFluenceActor::UserSteppingActionInVoxel(const int index, const G4Step* step)
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel - begin\n");
  const double weight = step->GetTrack()->GetWeight();
  /* Is this necessary? */
  if (index < 0)
    {
      GateDebugMessage("Actor", 5, "index<0 : do nothing\n");GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end\n");
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
      if (!mIgnoreWeight)
        {
          respValue *= weight;
        }

      bool sameEvent = true;
      if (mIsLastHitEventImageEnabled)
        {
          GateDebugMessage( "Actor", 2, "GateFluenceActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << Gateendl);
          if (mCurrentEvent != mLastHitEventImage.GetValue(index))
            {
              sameEvent = false;
              mLastHitEventImage.SetValue(index, mCurrentEvent);
            }
        }

      if (mIsUncertaintyImageEnabled || mIsSquaredImageEnabled)
        {
          if (sameEvent)
            {
              mImage.AddTempValue(index, respValue);
            }
          else
            {
              mImage.AddValueAndUpdate(index, respValue);
            }
        }
      else
        {
          mImage.AddValue(index, respValue);
        }

      if (mIsNumberOfHitsImageEnabled)
        {

          if (mIgnoreWeight)
            {
              mNumberOfHitsImage.AddValue(index, 1);
            }
          else
            {
              mNumberOfHitsImage.AddValue(index, weight);
            }
        }

      if (mIsScatterImageEnabled)
        {
          unsigned int order = 0;
          G4String process = "";
          /* Scatter order */
          if (info)
            {
              order = info->GetScatterOrder();
              process = info->GetScatterProcess();
              /* Allocate GateImage if process occurs */
              if (mProcesses.find(process) == mProcesses.end() && process != G4String(""))
                {
                  GateImage * voidImage = new GateImage;
                  voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
                  voidImage->Allocate();
                  voidImage->SetOrigin(mOrigin);
                  voidImage->Fill(0);
                  mProcesses.insert(std::pair<G4String, GateImage*>(process, voidImage));
                  mProcessName.push_back(process);
                }
              if (order)
                {
                  while (order > mFluencePerOrderImages.size() && order > 0)
                    {
                      GateImage * voidImage = new GateImage;
                      voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
                      voidImage->Allocate();
                      voidImage->SetOrigin(mOrigin);
                      voidImage->Fill(0);
                      mFluencePerOrderImages.push_back(voidImage);
                    }
                }
            }
          /* Scattered primary particles, e.g., primary photons that undergo
             Compton and Rayleigh interactions. Straight interactions are missed. */
          if (!step->GetTrack()->GetParentID()
              && !step->GetTrack()->GetDynamicParticle()->GetPrimaryParticle()->GetMomentum().isNear(step->GetTrack()->GetDynamicParticle()->GetMomentum()))
            {

              if (mIsUncertaintyImageEnabled || mIsSquaredImageEnabled)
                {
                  if (sameEvent)
                    {
                      mImageProcess.AddTempValue(index, respValue);
                    }
                  else
                    {
                      mImageProcess.AddValueAndUpdate(index, respValue);
                    }
                }
              else
                {
                  mImageProcess.AddValue(index, respValue);
                }

              if (process != G4String(""))
                {
                  mProcesses[process]->AddValue(index, respValue);
                }

              /* Scatter order image */
              if (order)
                {
                  mFluencePerOrderImages[order - 1]->AddValue(index, respValue);
                }
            }
          /* Secondary particles, e.g., Fluorescence gammas */
          if (step->GetTrack()->GetTrackID() && step->GetTrack()->GetParentID() > 0)
            {
              if (mIsUncertaintyImageEnabled || mIsSquaredImageEnabled)
                {
                  if (sameEvent)
                    {
                      mImageProcess.AddTempValue(index, respValue);
                    }
                  else
                    {
                      mImageProcess.AddValueAndUpdate(index, respValue);
                    }
                }
              else
                {
                  mImageProcess.AddValue(index, respValue);
                }

              /* Scatter order image */
              if (process != G4String(""))
                {
                  mProcesses[process]->AddValue(index, respValue);
                }
              if (order)
                {
                  mFluencePerOrderImages[order - 1]->AddValue(index, respValue);
                }
            }
        }
    }GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end\n");


  //void GateTrackLengthActor::PostUserTrackingAction(const GateVVolume * /*vol*/, const G4Track* aTrack)
  //{
  //aTrack->GetTrackLength(),aTrack->GetWeight();
  //}
}
