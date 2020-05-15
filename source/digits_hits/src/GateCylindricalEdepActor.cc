/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateCylindricalEdepActor :
  \brief
*/

// gate
#include "GateCylindricalEdepActor.hh"
#include "GateMiscFunctions.hh"

// g4
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateCylindricalEdepActor::GateCylindricalEdepActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateCylindricalEdepActor() -- begin\n");

  mCurrentEvent=-1;
  mIsEdepImageEnabled = false;  
  mIsDoseImageEnabled = false;
  mIsFluenceImageEnabled = false;  

  pMessenger = new GateCylindricalEdepActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateCylindricalEdepActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateCylindricalEdepActor::~GateCylindricalEdepActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/// Construct
void GateCylindricalEdepActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateCylindricalEdepActor -- Construct - begin\n");
  GateVImageActor::Construct();



  // Record the stepHitType
  mUserStepHitType = mStepHitType;

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);

  // Check if at least one image is enabled
  if (!mIsEdepImageEnabled && !mIsDoseImageEnabled &&!mIsFluenceImageEnabled)  {
    GateError("The CylindricalEdepActor " << GetObjectName()
              << " does not have any image enabled ...\n Please select at least one ('enableEdep true' for example)");
  }

  // Output Filename
  mEdepFilename = G4String(removeExtension(mSaveFilename))+"-Edep."+G4String(getExtension(mSaveFilename));
  mDoseFilename = G4String(removeExtension(mSaveFilename))+"-Dose."+G4String(getExtension(mSaveFilename));
  mFluenceFilename = G4String(removeExtension(mSaveFilename))+"-Fluence."+G4String(getExtension(mSaveFilename));
  
  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mDoseImage);
  SetOriginTransformAndFlagToImage(mFluenceImage);

  if (mIsEdepImageEnabled) {
    mEdepImage.SetResolutionAndHalfSizeCylinder(mResolution, mHalfSize, mPosition);
    mEdepImage.Allocate();
    mEdepImage.SetFilename(mEdepFilename);
  }
 
  if (mIsDoseImageEnabled) {
    mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mDoseImage.Allocate();
    mDoseImage.SetFilename(mDoseFilename);
    G4cout<< "allocate dose image and set resolution and file name" <<G4endl<<G4endl;
  }
 
  if (mIsFluenceImageEnabled) {
    mFluenceImage.SetResolutionAndHalfSizeCylinder(mResolution, mHalfSize, mPosition);
    mFluenceImage.Allocate();
    mFluenceImage.SetFilename(mFluenceFilename);
  }
  // Print information
  //GateMessage("Actor", 1,
              //"Dose DoseActor    = '" << GetObjectName() << "'\n" <<
              //"\tDose image        = " << mIsDoseImageEnabled << Gateendl <<
              //"\tDose squared      = " << mIsDoseSquaredImageEnabled << Gateendl <<
              //"\tDose uncertainty  = " << mIsDoseUncertaintyImageEnabled << Gateendl <<
              //"\tDose to water image        = " << mIsDoseToWaterImageEnabled << Gateendl <<
              //"\tDose to water squared      = " << mIsDoseToWaterSquaredImageEnabled << Gateendl <<
              //"\tDose to water uncertainty  = " << mIsDoseToWaterUncertaintyImageEnabled << Gateendl <<
              //"\tEdep image        = " << mIsEdepImageEnabled << Gateendl <<
              //"\tEdep squared      = " << mIsEdepSquaredImageEnabled << Gateendl <<
              //"\tEdep uncertainty  = " << mIsEdepUncertaintyImageEnabled << Gateendl <<
              //"\tNumber of hit     = " << mIsNumberOfHitsImageEnabled << Gateendl <<
              //"\t     (last hit)   = " << mIsLastHitEventImageEnabled << Gateendl <<
              //"\tDose algorithm    = " << mDoseAlgorithmType << Gateendl <<
              //"\tMass image (import) = " << mImportMassImage << Gateendl <<
              //"\tMass image (export) = " << mExportMassImage << Gateendl <<
              //"\tEdepFilename      = " << mEdepFilename << Gateendl <<
              //"\tDoseFilename      = " << mDoseFilename << Gateendl <<
              //"\tNb Hits filename  = " << mNbOfHitsFilename << Gateendl);

  ResetData();
  GateMessageDec("Actor", 4, "GateCylindricalEdepActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateCylindricalEdepActor::SaveData() {
  GateVActor::SaveData(); // (not needed because done into GateImageWithStatistic)

  if (mIsEdepImageEnabled) mEdepImage.SaveData(mCurrentEvent+1);
  if (mIsFluenceImageEnabled) mFluenceImage.SaveData(mCurrentEvent+1);
  if (mIsDoseImageEnabled) {
	  mDoseImage.SaveData(mCurrentEvent+1);
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCylindricalEdepActor::ResetData() {
  if (mIsEdepImageEnabled) mEdepImage.Reset();
  if (mIsFluenceImageEnabled) mFluenceImage.Reset();
  if (mIsDoseImageEnabled) mDoseImage.Reset();

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCylindricalEdepActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateCylindricalEdepActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);
  //
}
//-----------------------------------------------------------------------------
 

//-----------------------------------------------------------------------------
// Callback at each event
void GateCylindricalEdepActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateCylindricalEdepActor -- Begin of Event: "<<mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCylindricalEdepActor::UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track)
{
  if(track->GetDefinition()->GetParticleName() == "gamma") { mStepHitType = PostStepHitTypeCylindricalCS; }
  else { mStepHitType = RandomStepHitTypeCylindricalCS; }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCylindricalEdepActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateCylindricalEdepActor -- UserSteppingActionInVoxel - begin\n");
  GateDebugMessageInc("Actor", 4, "enedepo = " << step->GetTotalEnergyDeposit() << Gateendl);
  GateDebugMessageInc("Actor", 4, "weight = " <<  step->GetTrack()->GetWeight() << Gateendl);


	  const double weight = step->GetTrack()->GetWeight();

  // if energy is deposited outside image => do nothing

  if (index < 0) {
    GateDebugMessage("Actor", 5, "index < 0 : do nothing\n");
    GateDebugMessageDec("Actor", 4, "GateCylindricalEdepActor -- UserSteppingActionInVoxel -- end\n");
    return;
  }
    if (mIsFluenceImageEnabled)
    {
		
		 const double weightedsteplength = step->GetStepLength()*weight;
		mFluenceImage.AddValue(index, weightedsteplength);
    }
    
    
  if (mIsEdepImageEnabled || mIsDoseImageEnabled) {
	  
	  const double edep = step->GetTotalEnergyDeposit()/MeV*weight;//*step->GetTrack()->GetWeight();
  
	  // if no energy is deposited => do nothing
	  if (edep == 0) {
		GateDebugMessage("Actor", 5, "edep == 0 : do nothing\n");
		GateDebugMessageDec("Actor", 4, "GateCylindricalEdepActor -- UserSteppingActionInVoxel -- end\n");
		return;
	  }


	  double dose=0.;
	  if (mIsDoseImageEnabled) {
		  
		  ////---------------------------------------------------------------------------------
		  //// Volume weighting
		  double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
		  ////---------------------------------------------------------------------------------
		// ------------------------------------
		// Convert deposited energy into Gray
		dose = edep/density/gray;
		// ------------------------------------

		GateDebugMessage("Actor", 2,  "GateCylindricalEdepActor -- UserSteppingActionInVoxel:\tdose = "
				 << G4BestUnit(dose, "Dose")
				 << " rho = "
				 << G4BestUnit(density, "Volumic Mass")<< Gateendl );
		// here dose is added to voxel with index
		mDoseImage.AddValue(index, dose);
	  }
	  else if (mIsEdepImageEnabled)
		{
			mEdepImage.AddValue(index, edep);
		}
	}
  

  GateDebugMessageDec("Actor", 4, "GateCylindricalEdepActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
