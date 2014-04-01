/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateDoseActor :
  \brief
*/

#ifndef GATEDOSEACTOR_CC
#define GATEDOSEACTOR_CC

#include "GateDoseActor.hh"
#include "GateMiscFunctions.hh"
#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>

//-----------------------------------------------------------------------------
GateDoseActor::GateDoseActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateDoseActor() -- begin"<<G4endl);

  mCurrentEvent=-1;
  mIsEdepImageEnabled = false;
  mIsLastHitEventImageEnabled = false;
  mIsEdepSquaredImageEnabled = false;
  mIsEdepUncertaintyImageEnabled = false;
  mIsDoseImageEnabled = true;
  mIsDoseSquaredImageEnabled = false;
  mIsDoseUncertaintyImageEnabled = false;
  mIsDoseToWaterImageEnabled = false;
  mIsDoseToWaterSquaredImageEnabled = false;
  mIsDoseToWaterUncertaintyImageEnabled = false;
  mIsNumberOfHitsImageEnabled = false;
  mIsDoseNormalisationEnabled = false;
  mIsDoseToWaterNormalisationEnabled = false;
  mIsRBE1AlphaImageEnabled = false;
  mIsRBE1BetaImageEnabled = false;
  mIsRBE1FactorImageEnabled = false;
  mIsRBE1BioDoseImageEnabled = false;
  mIsRBE1Enabled = false;
  mIsRBE1Test1Enabled = false;
  //mIsRBE1DoseAveragedLetIsEnabled = false;
  mRBE1AlphaDataFilename = "pleaseSetAlphaDataFilename.txt";
  mRBE1AlphaDataFilename = "pleaseSetBetaDataFilename.txt";

  pMessenger = new GateDoseActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateDoseActor() -- end"<<G4endl);
  emcalc = new G4EmCalculator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateDoseActor::~GateDoseActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseActor::EnableDoseNormalisationToMax(bool b) {
  mIsDoseNormalisationEnabled = b;
  mDoseImage.SetNormalizeToMax(b);
  mDoseImage.SetScaleFactor(1.0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseActor::EnableDoseNormalisationToIntegral(bool b) {
  mIsDoseNormalisationEnabled = b;
  mDoseImage.SetNormalizeToIntegral(b);
  mDoseImage.SetScaleFactor(1.0);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateDoseActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateDoseActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct();

  // Find G4_WATER. This it needed here because we will used this
  // material for dedx computation for DoseToWater.
  G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");

  // Record the stepHitType
  mUserStepHitType = mStepHitType;

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);

  // Check if at least one image is enabled
  if (!mIsEdepImageEnabled &&
      !mIsDoseImageEnabled &&
      !mIsDoseToWaterImageEnabled &&
      !mIsNumberOfHitsImageEnabled &&
      !mIsRBE1AlphaImageEnabled &&
      !mIsRBE1BetaImageEnabled &&
      !mIsRBE1FactorImageEnabled &&
      !mIsRBE1BioDoseImageEnabled)  {
    GateError("The DoseActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableEdep true' for example)");
  }

  // Output Filename
  mEdepFilename = G4String(removeExtension(mSaveFilename))+"-Edep."+G4String(getExtension(mSaveFilename));
  mDoseFilename = G4String(removeExtension(mSaveFilename))+"-Dose."+G4String(getExtension(mSaveFilename));
  mDoseToWaterFilename = G4String(removeExtension(mSaveFilename))+"-DoseToWater."+G4String(getExtension(mSaveFilename));
  mNbOfHitsFilename = G4String(removeExtension(mSaveFilename))+"-NbOfHits."+G4String(getExtension(mSaveFilename));

  mRBE1AlphaFilename = G4String(removeExtension(mSaveFilename))+"-RBE1-Alpha."+G4String(getExtension(mSaveFilename));
  mRBE1BetaFilename = G4String(removeExtension(mSaveFilename))+"-RBE1-Beta."+G4String(getExtension(mSaveFilename));
  mRBE1FactorFilename = G4String(removeExtension(mSaveFilename))+"-RBE1-Factor."+G4String(getExtension(mSaveFilename));
  mRBE1BioDoseFilename = G4String(removeExtension(mSaveFilename))+"-RBE1-BioDose."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mDoseImage);
  SetOriginTransformAndFlagToImage(mNumberOfHitsImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);
  SetOriginTransformAndFlagToImage(mDoseToWaterImage);
  SetOriginTransformAndFlagToImage(mRBE1AlphaImage);
  SetOriginTransformAndFlagToImage(mRBE1AlphaImage);
  SetOriginTransformAndFlagToImage(mRBE1FactorImage);
  SetOriginTransformAndFlagToImage(mRBE1BioDoseImage);

  // Resize and allocate images
  if (mIsEdepSquaredImageEnabled || mIsEdepUncertaintyImageEnabled ||
      mIsDoseSquaredImageEnabled || mIsDoseUncertaintyImageEnabled ||
      mIsDoseToWaterSquaredImageEnabled || mIsDoseToWaterUncertaintyImageEnabled) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
  }
  if (mIsEdepImageEnabled) {
    //  mEdepImage.SetLastHitEventImage(&mLastHitEventImage);
    mEdepImage.EnableSquaredImage(mIsEdepSquaredImageEnabled);
    mEdepImage.EnableUncertaintyImage(mIsEdepUncertaintyImageEnabled);
    // Force the computation of squared image if uncertainty is enabled
    if (mIsEdepUncertaintyImageEnabled) mEdepImage.EnableSquaredImage(true);
    mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mEdepImage.Allocate();
    mEdepImage.SetFilename(mEdepFilename);
  }
  if (mIsDoseImageEnabled) {
    // mDoseImage.SetLastHitEventImage(&mLastHitEventImage);
    mDoseImage.EnableSquaredImage(mIsDoseSquaredImageEnabled);
    mDoseImage.EnableUncertaintyImage(mIsDoseUncertaintyImageEnabled);
    mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    // Force the computation of squared image if uncertainty is enabled
    if (mIsDoseUncertaintyImageEnabled) mDoseImage.EnableSquaredImage(true);

    // DD(mDoseImage.GetVoxelVolume());
    //mDoseImage.SetScaleFactor(1e12/mDoseImage.GetVoxelVolume());
    mDoseImage.Allocate();
    mDoseImage.SetFilename(mDoseFilename);
  }
  if (mIsDoseToWaterImageEnabled) {
    mDoseToWaterImage.EnableSquaredImage(mIsDoseToWaterSquaredImageEnabled);
    mDoseToWaterImage.EnableUncertaintyImage(mIsDoseToWaterUncertaintyImageEnabled);
    // Force the computation of squared image if uncertainty is enabled
    if (mIsDoseToWaterUncertaintyImageEnabled) mDoseToWaterImage.EnableSquaredImage(true);
    mDoseToWaterImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mDoseToWaterImage.Allocate();
    mDoseToWaterImage.SetFilename(mDoseToWaterFilename);
  }
  if (mIsNumberOfHitsImageEnabled) {
    mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mNumberOfHitsImage.Allocate();
  }

  // RBE1
  mIsRBE1Enabled =
    mIsRBE1AlphaImageEnabled ||
    mIsRBE1BetaImageEnabled ||
    mIsRBE1FactorImageEnabled ||
    mIsRBE1BioDoseImageEnabled;

  if (mIsRBE1Enabled || mIsRBE1Test1Enabled) {
    if (!mIsEdepImageEnabled) GateError("Please enable Edep sensor with RBE1." << G4endl);
    if (!mIsDoseImageEnabled) GateError("Please enable Dose sensor with RBE1." << G4endl);
    //DS TODO : no squared/uncertainty for the moment

    //    GateMessage("Actor", 0, " ALLOCATE RBE1 Image" << G4endl); //DS to remove
    mRBE1AlphaImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    //mRBE1AlphaImage.SetScaleFactor(1.0/mEdepImage.GetVoxelVolume());
    mRBE1AlphaImage.Allocate();
    mRBE1AlphaImage.SetFilename(mRBE1AlphaFilename);

    mRBE1BetaImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    //mRBE1BetaImage.SetScaleFactor(1.0/mEdepImage.GetVoxelVolume());
    mRBE1BetaImage.Allocate();
    mRBE1BetaImage.SetFilename(mRBE1BetaFilename);

    ReadRBE1AlphaBetaFromFile(mRBE1AlphaDataFilename, mRBE1BetaDataFilename);

    if (mIsRBE1FactorImageEnabled) {
      mRBE1FactorImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mRBE1FactorImage.Allocate();
      mRBE1FactorImage.SetFilename(mRBE1FactorFilename);
    }
    if (mIsRBE1BioDoseImageEnabled) {
      mRBE1BioDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mRBE1BioDoseImage.SetScaleFactor(1.0/mRBE1BioDoseImage.GetVoxelVolume());
      mRBE1BioDoseImage.Allocate();
      mRBE1BioDoseImage.SetFilename(mRBE1BioDoseFilename);
    }
    /*if (mIsRBE1DoseAveragedLetIsEnabled) {
      mRBE1DoseAVeragedLetImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      mRBE1DoseAVeragedLetImage.SetScaleFactor(1.0/mRBE1BioDoseImage.GetVoxelVolume());
      mRBE1DoseAVeragedLetImage.Allocate();
      mRBE1DoseAVeragedLetImage.SetFilename(mRBE1DoseAVeragedLetFilename);
      }*/

  }

  if (mIsRBE1Test1Enabled) mIsRBE1Enabled = false;

  // Print information
  GateMessage("Actor", 1,
              "\tDose DoseActor    = '" << GetObjectName() << "'" << G4endl <<
              "\tDose image        = " << mIsDoseImageEnabled << G4endl <<
              "\tDose squared      = " << mIsDoseSquaredImageEnabled << G4endl <<
              "\tDose uncertainty  = " << mIsDoseUncertaintyImageEnabled << G4endl <<
              "\tDose to water image        = " << mIsDoseToWaterImageEnabled << G4endl <<
              "\tDose to water squared      = " << mIsDoseToWaterSquaredImageEnabled << G4endl <<
              "\tDose to wateruncertainty  = " << mIsDoseToWaterUncertaintyImageEnabled << G4endl <<
              "\tEdep image        = " << mIsEdepImageEnabled << G4endl <<
              "\tEdep squared      = " << mIsEdepSquaredImageEnabled << G4endl <<
              "\tEdep uncertainty  = " << mIsEdepUncertaintyImageEnabled << G4endl <<
              "\tNumber of hit     = " << mIsNumberOfHitsImageEnabled << G4endl <<
              "\t     (last hit)   = " << mIsLastHitEventImageEnabled << G4endl <<
              "\tRBE1 alpha        = " << mIsRBE1AlphaImageEnabled << G4endl <<
              "\tRBE1 beta         = " << mIsRBE1BetaImageEnabled << G4endl <<
              "\tRBE1 factor       = " << mIsRBE1FactorImageEnabled << G4endl <<
              "\tRBE1 biodose      = " << mIsRBE1BioDoseImageEnabled << G4endl <<
              "\tedepFilename      = " << mEdepFilename << G4endl <<
              "\tdoseFilename      = " << mDoseFilename << G4endl <<
              "\tRBE1AlphaFilename = " << mRBE1AlphaFilename << G4endl <<
              "\tRBE1BetaFilename  = " << mRBE1BetaFilename << G4endl <<
              "\tNb Hits filename  = " << mNbOfHitsFilename << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateDoseActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateDoseActor::SaveData() {
  GateVActor::SaveData(); // (not needed because done into GateImageWithStatistic)

  if (mIsEdepImageEnabled) mEdepImage.SaveData(mCurrentEvent+1);
  if (mIsDoseImageEnabled) {
    if (mIsDoseNormalisationEnabled)
      mDoseImage.SaveData(mCurrentEvent+1, true);
    else
      mDoseImage.SaveData(mCurrentEvent+1, false);
  }

  if (mIsDoseToWaterImageEnabled) {
    if (mIsDoseToWaterNormalisationEnabled)
      mDoseToWaterImage.SaveData(mCurrentEvent+1, true);
    else
      mDoseToWaterImage.SaveData(mCurrentEvent+1, false);
  }

  if (mIsLastHitEventImageEnabled) {
    mLastHitEventImage.Fill(-1); // reset
  }

  if (mIsNumberOfHitsImageEnabled) {
    mNumberOfHitsImage.Write(mNbOfHitsFilename);
  }
  if( mIsRBE1Enabled || mIsRBE1Test1Enabled) {
    if (mIsRBE1AlphaImageEnabled) mRBE1AlphaImage.SaveData(mCurrentEvent+1);
    if (mIsRBE1BetaImageEnabled) mRBE1BetaImage.SaveData(mCurrentEvent+1);
    if (mIsRBE1FactorImageEnabled || mIsRBE1BioDoseImageEnabled) ComputeRBE1ImageAndSave();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::ResetData() {
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsEdepImageEnabled) mEdepImage.Reset();
  if (mIsDoseImageEnabled) mDoseImage.Reset();
  if (mIsDoseToWaterImageEnabled) mDoseToWaterImage.Reset();
  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.Fill(0);
  if (mIsRBE1Enabled || mIsRBE1Test1Enabled) {
    mRBE1AlphaImage.Fill(0.0);
    mRBE1BetaImage.Fill(0.0);
    if (mIsRBE1FactorImageEnabled) mRBE1FactorImage.Reset(1.0);
    if (mIsRBE1BioDoseImageEnabled) mRBE1BioDoseImage.Reset();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateDoseActor -- Begin of Run" << G4endl);
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateDoseActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateDoseActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track)
{
  if(track->GetDefinition()->GetParticleName() == "gamma") { mStepHitType = PostStepHitType; }
  else { mStepHitType = mUserStepHitType; }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateDoseActor -- UserSteppingActionInVoxel - begin" << G4endl);
  GateDebugMessageInc("Actor", 4, "enedepo = " << step->GetTotalEnergyDeposit() << G4endl);
  GateDebugMessageInc("Actor", 4, "weight = " <<  step->GetTrack()->GetWeight() << G4endl);
  const double weight = step->GetTrack()->GetWeight();
  const double edep = step->GetTotalEnergyDeposit()*weight;//*step->GetTrack()->GetWeight();

  // if no energy is deposited or energy is deposited outside image => do nothing
  if (step->GetTotalEnergyDeposit() == 0) {
    GateDebugMessage("Actor", 5, "edep == 0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateDoseActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }
  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateDoseActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  // compute sameEvent
  // sameEvent is false the first time some energy is deposited for each primary particle
  bool sameEvent=true;
  if (mIsLastHitEventImageEnabled) {
    GateDebugMessage("Actor", 2,  "GateDoseActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
    if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
      sameEvent = false;
      mLastHitEventImage.SetValue(index, mCurrentEvent);
    }
  }

  double dose=0.;

  if (mIsDoseImageEnabled) {
    double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();

    // ------------------------------------
    // Convert deposited energy into Gray

    // OLD version (correct but not clear)
    // dose = edep/density*1e12/mDoseImage.GetVoxelVolume();

    // NEW version (same results but more clear)
    dose = edep/density/mDoseImage.GetVoxelVolume()/gray;
    // ------------------------------------

    GateDebugMessage("Actor", 2,  "GateDoseActor -- UserSteppingActionInVoxel:\tdose = "
		     << G4BestUnit(dose, "Dose")
		     << " rho = "
		     << G4BestUnit(density, "Volumic Mass")<<G4endl );
  }

  double doseToWater = 0;
  if (mIsDoseToWaterImageEnabled) {

    // to get nuclear inelastic cross-section, see "geant4.9.4.p01/examples/extended/hadronic/Hadr00/"
    // #include "G4HadronicProcessStore.hh"
    // G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
    // store->GetInelasticCrossSectionPerAtom(particle,e,elm);

    double cut = DBL_MAX;
    cut=1;
    double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();
    G4String material = step->GetPreStepPoint()->GetMaterial()->GetName();
    double Energy = step->GetPreStepPoint()->GetKineticEnergy();
    G4String PartName = step->GetTrack()->GetDefinition()->GetParticleName();
    //    const G4ParticleDefinition * PartDef = step->GetTrack()->GetParticleDefinition();
    //    G4Material  * MatDef = step->GetTrack()->GetMaterial();
    double DEDX=0, DEDX_Water=0;
    //    G4cout<<PartName<<"\t";//G4endl;//"  "<<edep<<"  "<<NonIonizingEdep<<G4endl;


    // Dose to water: it could be possible to make this process more
    // generic by choosing any material in place of water


    // Other particles should be taken into account (Helium etc), but bug ? FIXME
    if (PartName== "proton" || PartName== "e-" || PartName== "e+" || PartName== "deuteron"){
      //if (PartName != "O16[0.0]" && PartName != "alpha" && PartName != "Be7[0.0]" && PartName != "C12[0.0]"){

      DEDX = emcalc->ComputeTotalDEDX(Energy, PartName, material, cut);
      DEDX_Water = emcalc->ComputeTotalDEDX(Energy, PartName, "G4_WATER", cut);

      doseToWater=edep/density*1e12/mDoseToWaterImage.GetVoxelVolume()*(DEDX_Water/1.)/(DEDX/(density*1.6e-19));

    }
    else {
      DEDX = emcalc->ComputeTotalDEDX(100, "proton", material, cut);
      DEDX_Water = emcalc->ComputeTotalDEDX(100, "proton", "G4_WATER", cut);
      doseToWater=edep/density*1e12/mDoseToWaterImage.GetVoxelVolume()*(DEDX_Water/1.)/(DEDX/(density*1.6e-19));
    }

    GateDebugMessage("Actor", 2,  "GateDoseActor -- UserSteppingActionInVoxel:\tdose to water = "
		     << G4BestUnit(doseToWater, "Dose to water")
		     << " rho = "
		     << G4BestUnit(density, "Volumic Mass")<<G4endl );
  }


  if (mIsEdepImageEnabled) {
    GateDebugMessage("Actor", 2, "GateDoseActor -- UserSteppingActionInVoxel:\tedep = " << G4BestUnit(edep, "Energy") << G4endl);
  }



  if (mIsDoseImageEnabled) {

    if (mIsDoseUncertaintyImageEnabled || mIsDoseSquaredImageEnabled) {
      if (sameEvent) mDoseImage.AddTempValue(index, dose);
      else mDoseImage.AddValueAndUpdate(index, dose);
    }
    else mDoseImage.AddValue(index, dose);
  }

  if (mIsDoseToWaterImageEnabled) {

    if (mIsDoseToWaterUncertaintyImageEnabled || mIsDoseToWaterSquaredImageEnabled) {
      if (sameEvent) mDoseToWaterImage.AddTempValue(index, doseToWater);
      else mDoseToWaterImage.AddValueAndUpdate(index, doseToWater);
    }
    else mDoseToWaterImage.AddValue(index, doseToWater);
  }

  if (mIsEdepImageEnabled) {
    if (mIsEdepUncertaintyImageEnabled || mIsEdepSquaredImageEnabled) {
      if (sameEvent) mEdepImage.AddTempValue(index, edep);
      else mEdepImage.AddValueAndUpdate(index, edep);
    }
    else mEdepImage.AddValue(index, edep);
  }

  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.AddValue(index, weight);


  // Biological dose
  if (mIsRBE1Enabled) {
    G4double length = step->GetStepLength();
    G4double let = (edep/keV)/(length/um);
    GateDebugMessage("Actor", 2, "\tlg  = " << G4BestUnit(length, "Length")  << G4endl);
    GateDebugMessage("Actor", 2, "\tlet = " << G4BestUnit(let, "Energy/Length") << G4endl);
    GateDebugMessage("Actor", 2, "\tlet = " << let/(keV/um) << " KeV/um" << G4endl);
    GateDebugMessage("Actor", 2, "\tlet = " << let << "" << G4endl);

    // um = micrometer
    G4double alpha=0, beta=0;
    GetRBE1AlphaBetaFromLet(let, alpha, beta);
    GateDebugMessage("Actor", 2, "\talpha = " << alpha << "\t beta = " << beta << G4endl);
    GateDebugMessage("Actor", 2, "\talpha*Edep = " << alpha*edep << "\t sqrt(beta)*Edep = " << sqrt(beta)*edep << G4endl);
    mRBE1AlphaImage.AddValue(index, alpha*edep);
    mRBE1BetaImage.AddValue(index, sqrt(beta)*edep);
    /*if (mIsRBE1DoseAveragedLetIsEnabled)
      mRBE1DoseAVeragedLetImage.addvalue(index, edep*let);*/
  }

  if (mIsRBE1Test1Enabled) {
    G4ParticleDefinition * p = step->GetTrack()->GetDefinition();
    G4String particleName = p->GetParticleName();
    //    if (particleName == "C12[0.0]") {
    G4double length = step->GetStepLength();
    G4double let = (edep/keV)/(length/um);
    G4double alpha=0, beta=0;
    GetRBE1AlphaBetaFromLet(let, alpha, beta);

    /*
      GateMessage("Actor", 0, "particle = " << particleName
      //		<< " mass = " << p->GetPDGMass()
      //<< " PDGwidth = " << p->GetPDGWidth()
      //<< " PDGCharge = " << p->GetPDGCharge()
      << " atom nb = " << p->GetAtomicNumber()
      << " atom mass = " << p->GetAtomicMass()
      << " lg(um) = " << length/um
      << " edep(keV) =" << edep/keV
      << " let= " << let
      << " a = " << alpha
      << G4endl);
    */

    if (p->GetAtomicNumber() == 6) {
      mRBE1AlphaImage.AddValue(index, alpha*edep);
      mRBE1BetaImage.AddValue(index, sqrt(beta)*edep);
      /*if (mIsRBE1DoseAveragedLetIsEnabled)
	mRBE1DoseAVeragedLetImage.addvalue(index, edep*let);*/
    }
  }

  GateDebugMessageDec("Actor", 4, "GateDoseActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::ReadRBE1AlphaBetaFromFile(G4String filenameAlpha, G4String filenameBeta) {
  std::ifstream is;
  OpenFileInput(filenameAlpha, is);
  skipComment(is);
  double l;
  while (is) {
    is >> l; mAlphaLet.push_back(l);
    is >> l; mAlphaValues.push_back(l);
  }
  is.close();
  std::ifstream is2;
  G4cout << "Read [" << filenameBeta << "]" << G4endl;
  OpenFileInput(filenameBeta, is2);
  skipComment(is2);
  while (is2) {
    is2 >> l; mBetaLet.push_back(l);
    is2 >> l; mBetaValues.push_back(l);
  }
  is2.close();
  // Debug
  for(unsigned int i=0; i<mAlphaLet.size(); i++)
    GateDebugMessage("Event", 1, i << " = " << mAlphaLet[i] << " " << mAlphaValues[i] << G4endl);
  for(unsigned int i=0; i<mBetaLet.size(); i++)
    GateDebugMessage("Event", 1, i << " = " << mBetaLet[i] << " " << mBetaValues[i] << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::GetRBE1AlphaBetaFromLet(G4double let, G4double & alpha, G4double & beta) {
  //DS BIDON juste pour essai ...
  alpha = LinearInterpolation(let, mAlphaLet, mAlphaValues);
  beta = LinearInterpolation(let, mBetaLet, mBetaValues);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseActor::ComputeRBE1ImageAndSave() {

  GateDebugMessage("Actor", 3, "ComputeRBE1ImageAndSave" << G4endl);

  GateImage::iterator       po = mRBE1FactorImage.GetValueImage().begin();
  GateImage::const_iterator pe = mRBE1FactorImage.GetValueImage().end();
  GateImage::const_iterator pa = mRBE1AlphaImage.GetValueImage().begin();
  GateImage::const_iterator pb = mRBE1BetaImage.GetValueImage().begin();
  GateImage::const_iterator pedep = mEdepImage.GetValueImage().begin();

  // See Kase2006
  // http://www.iop.org/EJ/abstract/0031-9155/51/24/N03
  G4double alphaBar, betaBar, edep, alpha, beta, rbe;
  while (po != pe) {
    edep = (*pedep);
    alpha = (*pa);
    beta = (*pb);

    if (*pedep != 0 && (alpha !=0 && beta!=0) ) {

      alphaBar = alpha/edep;
      betaBar = beta/edep;
      rbe = (4.08*2*betaBar)/(-alphaBar+sqrt(alphaBar*alphaBar-4*betaBar*log(0.1)));
      //DD(log(0.1)); = log(0.1) = [ -2.30259 ]
      // Note that, here, log is natural logarithm, denoted ln

      // debug output
      //       GateDebugMessage("Actor", 1, "RBE1 edep=" << edep
      // 		       << " a=" << alpha << " b=" << beta
      // 		       << " ab=" << alphaBar << " bb=" << betaBar
      // 		       << " ==> rbe = " << rbe << G4endl);
      // Store rbe
      *po = rbe;
    }
    else *po = 1.0;
    ++po;
    ++pa;
    ++pb;
    ++pedep;
  }

  if (mIsRBE1FactorImageEnabled) mRBE1FactorImage.SaveData(mCurrentEvent+1);

  if (mIsRBE1BioDoseImageEnabled) {
    GateImage::iterator       po = mRBE1BioDoseImage.GetValueImage().begin();
    GateImage::const_iterator pe = mRBE1BioDoseImage.GetValueImage().end();
    GateImage::const_iterator pd = mDoseImage.GetValueImage().begin();
    GateImage::const_iterator prbe = mRBE1FactorImage.GetValueImage().begin();
    while (po != pe) {
      *po = (*prbe) * (*pd);
      GateDebugMessage("Actor", 6, "rbe=" << (*prbe) << " d=" << (*pd) << " => " << *po << G4endl);
      ++po;
      ++pd;
      ++prbe;
    }
    mRBE1BioDoseImage.SaveData(mCurrentEvent+1);
  }

}
//-----------------------------------------------------------------------------

#endif /* end #define GATEDOSEACTOR_CC */
