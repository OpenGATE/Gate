/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateLETActor :
  \brief
*/

// gate
#include "GateLETActor.hh"
#include "GateMiscFunctions.hh"

// g4
#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateLETActor::GateLETActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateLETActor() -- begin\n");

  mCurrentEvent=-1;
  mIsRestrictedFlag = false;
  mIsTrackAveragedFluenceAveraged=false;
  mIsTrackAveragedDXAveraged=false;
  mIsTrackAveragedDXAveragedCancelled=false;
  mIsDoseAveragedDEDXAveraged=false;
  mIsTrackAveragedFluenceTrackAveraged = false;
  mIsDoseAveragedEdepDXAveraged=false;

  mIsRelUncertEnabled = false;
  //mDeltaRestricted = 800000.0;
  mIsLETUncertaintyImageEnabled = false;
  mIsDoseToWaterEnabled = false;
  mIsParallelCalculationEnabled = false;
  mAveragingType = "DoseAveraged";
  pMessenger = new GateLETActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateLETActor() -- end\n");
  emcalc = new G4EmCalculator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateLETActor::~GateLETActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateLETActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateLETActor -- Construct - begin\n");
  GateVImageActor::Construct();

  // Find G4_WATER. This it needed here because we will used this
  // material for dedx computation for DoseToWater.
  G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);


  if (mAveragingType == "DoseAveraged" || mAveragingType == "DoseAverage" || mAveragingType == "doseaverage" || mAveragingType == "dose"){mIsDoseAveragedDEDXAveraged = true;}
  else if (mAveragingType == "TrackAveragedFluenceStep"){mIsTrackAveragedFluenceAveraged = true;}
  else if (mAveragingType == "TrackAveragedCancelled"){mIsTrackAveragedDXAveragedCancelled = true;}
  else if (mAveragingType == "TrackAveragedFluenceTrack"){mIsTrackAveragedFluenceTrackAveraged = true;}
  else if (mAveragingType == "TrackAveraged" || mAveragingType == "TrackAverage" || mAveragingType == "Track" || mAveragingType == "track" || mAveragingType == "TrackAveragedDXAveraged"){mIsTrackAveragedDXAveraged = true;}
  else if (mAveragingType == "DoseAveragedEdep"){mIsDoseAveragedEdepDXAveraged = true;}

  else {GateError("The LET averaging Type" << GetObjectName()
                  << " is not valid ...\n Please select 'DoseAveraged' or 'TrackAveraged')");}

  // Output Filename
  mLETFilename = mSaveFilename;
  if (mIsDoseAveragedDEDXAveraged)
    {
      mLETFilename= removeExtension(mSaveFilename) + "-doseAveraged."+ getExtension(mSaveFilename);
    }
  else if (mIsTrackAveragedDXAveraged)
    {
      mLETFilename= removeExtension(mSaveFilename) + "-trackAveraged."+ getExtension(mSaveFilename);
    }
  if (mIsDoseToWaterEnabled){
    mLETFilename= removeExtension(mLETFilename) + "-letToWater."+ getExtension(mLETFilename);
  }
  if (mIsParallelCalculationEnabled)
    {
      numeratorFileName= removeExtension(mLETFilename) + "-numerator."+ getExtension(mLETFilename);
      denominatorFileName= removeExtension(mLETFilename) + "-denominator."+ getExtension(mLETFilename);
    }

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mLETImage);
  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mFinalImage);
  SetOriginTransformAndFlagToImage(mLETSecondMomentImage);
  SetOriginTransformAndFlagToImage(mLETUncertaintyFinalImage);
  SetOriginTransformAndFlagToImage(mRelUncertImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);
  SetOriginTransformAndFlagToImage(mLETTempImage);
  SetOriginTransformAndFlagToImage(mNumberOfHitsImage);

  // Resize and allocate images
  mLETImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLETImage.Allocate();
  mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mEdepImage.Allocate();
  mFinalImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mFinalImage.Allocate();
  if (mIsRelUncertEnabled){
    mRelUncertImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mRelUncertImage.Allocate();
  }
  if (mIsTrackAveragedFluenceTrackAveraged) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();

    mLETTempImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLETTempImage.Allocate();

    mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mNumberOfHitsImage.Allocate();
  }

  if (mIsLETUncertaintyImageEnabled) {
    mLETSecondMomentImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLETSecondMomentImage.Allocate();
    mLETUncertaintyFinalImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLETUncertaintyFinalImage.Allocate();
  }

  // Warning: for the moment we force to PostStepHitType. This is ok
  // (slightly faster) if voxel sizes are the same between the
  // let-actor and the attached voxelized volume. But wrong if not.
  mStepHitType = PostStepHitType;// RandomStepHitType; // Warning

  // Print information
  GateMessage("Actor", 1,
              "\tLET Actor      = '" << GetObjectName() << Gateendl <<
              "\tLET image      = " << mLETFilename << Gateendl <<
              "\tResolution     = " << mResolution << Gateendl <<
              "\tHalfSize       = " << mHalfSize << Gateendl <<
              "\tPosition       = " << mPosition << Gateendl);

  ResetData();
  GateMessageDec("Actor", 4, "GateLETActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save datamDeltaRestricted
void GateLETActor::SaveData() {
  GateVActor::SaveData();
  if (mIsRelUncertEnabled){
    mRelUncertImage.Write(removeExtension(mLETFilename) + "-relUncert."+ getExtension(mLETFilename));
  }

  if (mIsParallelCalculationEnabled) {
    mLETImage.Write(numeratorFileName);
    mEdepImage.Write(denominatorFileName);
    if (mIsLETUncertaintyImageEnabled) {
      mLETSecondMomentImage.Write(removeExtension(mLETFilename) + "-variance-unnormalizedSecondMoment."+ getExtension(mLETFilename));
    }
  }
  else
    {
      GateImageDouble::const_iterator iter_LET = mLETImage.begin();
      GateImageDouble::const_iterator iter_Edep = mEdepImage.begin();
      GateImageDouble::iterator iter_Final = mFinalImage.begin();
      for(iter_LET = mLETImage.begin(); iter_LET != mLETImage.end(); iter_LET++) {
        if (*iter_Edep == 0.0) *iter_Final = 0.0; // do not divide by zero
        else *iter_Final = (*iter_LET)/(*iter_Edep);
        iter_Edep++;
        iter_Final++;
      }
      mFinalImage.Write(mLETFilename);

      uncertaintyFilename = removeExtension(mLETFilename) + "-variance."+ getExtension(mLETFilename);
      if (mIsLETUncertaintyImageEnabled)
        {
	  GateImageDouble::const_iterator iter_LET_u = mLETImage.begin();
	  GateImageDouble::const_iterator iter_LET_secMoment = mLETSecondMomentImage.begin();
	  GateImageDouble::const_iterator iter_Edep_u = mEdepImage.begin();
	  GateImageDouble::iterator iter_Final_uncert = mLETUncertaintyFinalImage.begin();
	  for(iter_LET_u = mLETImage.begin(); iter_LET_u != mLETImage.end(); iter_LET_u++) {
            if (*iter_Edep_u == 0.0) *iter_Final_uncert = 0.0; // do not divide by zero
            else *iter_Final_uncert = (*iter_LET_secMoment)/(*iter_Edep_u) - (*iter_LET_u)*(*iter_LET_u)/(*iter_Edep_u)/(*iter_Edep_u);
            iter_Edep_u++;
            iter_LET_secMoment++;
            iter_Final_uncert++;
	  }
	  mLETUncertaintyFinalImage.Write(uncertaintyFilename);
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::ResetData() {
  mLETImage.Fill(0.0);
  mEdepImage.Fill(0.0);

  if (mIsLETUncertaintyImageEnabled) {
    mLETSecondMomentImage.Fill(0.0);
  }

  if (mIsRelUncertEnabled) {
    mRelUncertImage.Fill(0.0);
  }

  if (mIsTrackAveragedFluenceTrackAveraged) {
    mLETTempImage.Fill(0.0);
    mNumberOfHitsImage.Fill(0);
    mLastHitEventImage.Fill(0);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateLETActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback at each event
void GateLETActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateLETActor -- Begin of Event: " << mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel - begin\n");
  GateDebugMessageInc("Actor", 4, "enedepo = " << step->GetTotalEnergyDeposit() << Gateendl);
  GateDebugMessageInc("Actor", 4, "weight = " <<  step->GetTrack()->GetWeight() << Gateendl);

  // Get edep and current particle weight
  const double weight = step->GetTrack()->GetWeight();

  // A.Resch tested calculation method:
  const double edep = step->GetTotalEnergyDeposit()*weight;

  double steplength = step->GetStepLength();

  //if no energy is deposited or energy is deposited outside image => do nothing
  if (edep == 0) {
    GateDebugMessage("Actor", 5, "GateLETActor edep == 0 : do nothing\n");
    return;
  }
  if (index < 0) {
    GateDebugMessage("Actor", 5, "GateLETActor pixel index < 0 : do nothing\n");
    return;
  }

  const G4Material* material = step->GetPreStepPoint()->GetMaterial();//->GetName();
  double energy1 = step->GetPreStepPoint()->GetKineticEnergy();
  double energy2 = step->GetPostStepPoint()->GetKineticEnergy();
  double energy=(energy1+energy2)/2;
  const G4ParticleDefinition* partname = step->GetTrack()->GetDefinition();//->GetParticleName();

  // Compute the dedx for the current particle in the current material
  double doseAveragedLET =0;
  G4double dedx = emcalc->ComputeElectronicDEDX(energy, partname, material);

  if (mIsRestrictedFlag) {
    // PDGcode 11 is an electron
    // LET restricted is defined in that way that all secondary electrons above a threshold are removed (they carry energy away)
    // see ICRU 85
    if (partname->GetPDGEncoding() == 11 && step->GetTrack()->GetParentID() == 0)
      {
        dedx = emcalc->ComputeElectronicDEDX(energy, partname, material,mDeltaRestricted);
      }
  }

  double normalizationVal = 0;
  if (mIsDoseAveragedDEDXAveraged) {
    doseAveragedLET=edep*dedx; // /(density/(g/cm3));
    normalizationVal = edep;
  }
  else if (mIsTrackAveragedDXAveraged) {
    doseAveragedLET=dedx*steplength;
    normalizationVal = steplength;
  }
  else if (mIsTrackAveragedFluenceTrackAveraged){
    if ( mLastHitEventImage.GetValue(index) == mCurrentEvent){
      mLETTempImage.AddValue(index, dedx);
      mNumberOfHitsImage.AddValue(index, 1);
    }
    else {
      if (mNumberOfHitsImage.GetValue(index) > 0) {
        doseAveragedLET = (double) mLETTempImage.GetValue(index)/mNumberOfHitsImage.GetValue(index);
      }
      else { doseAveragedLET = 0.0; }

      if (doseAveragedLET > 0.0001) {
        mLETImage.AddValue(index, doseAveragedLET);
        // Store the Edep (needed for final computation)
        mEdepImage.AddValue(index, 1.0);
      }
      mLETTempImage.SetValue(index,dedx);
      mNumberOfHitsImage.SetValue(index,1);
    }
    mLastHitEventImage.SetValue(index, mCurrentEvent);
  }
  else if (mIsTrackAveragedFluenceAveraged) {
    // this implementation probably varies with production threshold:
    doseAveragedLET = weight*(dedx);
    normalizationVal = weight;
  }
  else if (mIsTrackAveragedDXAveragedCancelled) {
    doseAveragedLET=edep;
    normalizationVal = steplength;
  }
  else if (mIsDoseAveragedEdepDXAveraged) {
    doseAveragedLET=edep*edep/steplength;
    normalizationVal = edep;
  }

  if (mIsDoseToWaterEnabled){
    doseAveragedLET=edep*emcalc->ComputeTotalDEDX(energy, partname->GetParticleName(), "G4_WATER");
  }

  if (mIsLETUncertaintyImageEnabled) {
    double secondMomentLET = 0;
    if (mIsDoseAveragedDEDXAveraged){
      secondMomentLET = edep*dedx*dedx;
    }
    else if (mIsTrackAveragedDXAveraged) { secondMomentLET = steplength*dedx*dedx;}
    else if (mIsTrackAveragedFluenceAveraged) { secondMomentLET = dedx*dedx;}
    else if (mIsTrackAveragedDXAveragedCancelled) { secondMomentLET = edep*edep/steplength;}
    else if (mIsDoseAveragedEdepDXAveraged) { secondMomentLET = edep*edep/steplength;}
    mLETSecondMomentImage.AddValue(index, secondMomentLET);
  }

  if (mIsRelUncertEnabled) {
    if (doseAveragedLET > 0) {
      double L_i = mLETImage.GetValue(index);
      double n_i = mEdepImage.GetValue(index);
      if (n_i>0){
        double relUncert = 1-(L_i/(L_i+doseAveragedLET)) *((n_i+normalizationVal)/n_i) ;
        double oldRelUncertVal = mRelUncertImage.GetValue(index);
        if (mCurrentEvent % 10000 == 0) {
          mRelUncertImage.SetValue(index, relUncert);
        }
        else if (std::abs(oldRelUncertVal) < relUncert) {
          mRelUncertImage.SetValue(index, relUncert);
        }
      }
    }
  }

  // Store the LET
  if (!mIsTrackAveragedFluenceTrackAveraged) {
    mLETImage.AddValue(index, doseAveragedLET);
    // Store the Edep (needed for final computation)
    mEdepImage.AddValue(index, normalizationVal);
  }

  GateDebugMessageDec("Actor", 4, "GateLETActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
