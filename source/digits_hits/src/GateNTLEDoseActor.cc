/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateNTLEDoseActor :
  \brief
*/

#include "GateNTLEDoseActor.hh"
#include "GateMiscFunctions.hh"

#include <G4PhysicalConstants.hh>

#include <TCanvas.h>

//-----------------------------------------------------------------------------
GateNTLEDoseActor::GateNTLEDoseActor(G4String name, G4int depth):
  GateVImageActor(name, depth) {
  mCurrentEvent = -1;
  pMessenger = new GateNTLEDoseActorMessenger(this);
  mKFHandler = new GateKermaFactorHandler();

  mIsDoseImageEnabled            = false;
  mIsDoseSquaredImageEnabled     = false;
  mIsDoseUncertaintyImageEnabled = false;

  mIsDoseCorrectionEnabled       = false;
  mIsDoseCorrectionTLEEnabled    = false;

  mIsLastHitEventImageEnabled    = false;
  mIsKermaFactorDumped           = false;
  mIsKillSecondaryEnabled        = false;

  mIsFluxImageEnabled            = false;
  mIsFluxSquaredImageEnabled     = false;
  mIsFluxUncertaintyImageEnabled = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNTLEDoseActor::~GateNTLEDoseActor() {
  delete pMessenger;

  if(mIsKermaFactorDumped)
    delete mg;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::Construct() {
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  if (!mIsDoseImageEnabled)
    GateError("The NTLEDoseActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableDose true' for example)");

  // Output Filename
  mDoseFilename = G4String(removeExtension(mSaveFilename)) + "-Dose." + G4String(getExtension(mSaveFilename));
  mFluxFilename = G4String(removeExtension(mSaveFilename)) + "-Flux." + G4String(getExtension(mSaveFilename));

  SetOriginTransformAndFlagToImage(mDoseImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);

  if (mIsDoseSquaredImageEnabled || mIsDoseUncertaintyImageEnabled) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
    mLastHitEventImage.SetOrigin(mOrigin);
  }

  if (mIsDoseImageEnabled) {
    mDoseImage.EnableSquaredImage(mIsDoseSquaredImageEnabled);
    mDoseImage.EnableUncertaintyImage(mIsDoseUncertaintyImageEnabled);
    mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mDoseImage.Allocate();
    mDoseImage.SetFilename(mDoseFilename);
    mDoseImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
    mDoseImage.SetOrigin(mOrigin);
  }

  if (mIsFluxImageEnabled) {
    mFluxImage.EnableSquaredImage    (mIsFluxSquaredImageEnabled);
    mFluxImage.EnableUncertaintyImage(mIsFluxUncertaintyImageEnabled);
    mFluxImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mFluxImage.Allocate();
    mFluxImage.SetFilename(mFluxFilename);
    mFluxImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
    mFluxImage.SetOrigin(mOrigin);
  }

  if (mIsKermaFactorDumped)
  {
    mg = new TMultiGraph();
    mg->SetTitle(";Neutron energy [MeV];Kerma factor [Gy*m^{2}/neutron]");
  }

  GateMessage("Actor", 1,
              "NTLE DoseActor    = '" << GetObjectName() << "'\n" <<
              "\tDose image        = " << mIsDoseImageEnabled << Gateendl <<
              "\tDose squared      = " << mIsDoseSquaredImageEnabled << Gateendl <<
              "\tDose uncertainty  = " << mIsDoseUncertaintyImageEnabled << Gateendl <<
              "\tDose correction   = " << mIsDoseCorrectionEnabled << Gateendl <<
              "\tDump kerma factor = " << mIsKermaFactorDumped << Gateendl <<
              "\tDoseFilename      = " << mDoseFilename << Gateendl <<
              "\tFlux image        = " << mIsFluxImageEnabled << Gateendl <<
              "\tFlux squared      = " << mIsFluxSquaredImageEnabled << Gateendl <<
              "\tFlux uncertainty  = " << mIsFluxUncertaintyImageEnabled << Gateendl <<
              "\tFluxFilename      = " << mFluxFilename << Gateendl);

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::SaveData() {
  GateVActor::SaveData();
  if (mIsDoseImageEnabled) mDoseImage.SaveData(mCurrentEvent + 1, false);
  if (mIsFluxImageEnabled) mFluxImage.SaveData(mCurrentEvent + 1, false);
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if(mIsKermaFactorDumped)
  {
    TCanvas* c = new TCanvas();
    c->SetLogx();
    c->SetLogy();
    c->SetGrid();
    mg->Draw("AP");
    c->SaveAs((removeExtension(mSaveFilename) + "-KermaFactorDump.root").c_str());
    c->SaveAs((removeExtension(mSaveFilename) + "-KermaFactorDump.eps").c_str());
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::ResetData() {
  if (mIsDoseImageEnabled) mDoseImage.Reset();
  if (mIsFluxImageEnabled) mFluxImage.Reset();
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsKermaFactorDumped)
  {
    mMaterialList.clear();
    mg->Clear();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::UserSteppingAction(const GateVVolume*, const G4Step* step) {
  const int index = GetIndexFromStepPosition(GetVolume(), step);
  UserSteppingActionInVoxel(index, step);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::BeginOfRunAction(const G4Run* r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateNTLEDoseActor -- Begin of Run\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::BeginOfEventAction(const G4Event* e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateNTLEDoseActor -- Begin of Event: "<< mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  if ( step->GetTrack()->GetDefinition()->GetParticleName() == "neutron" ||
      (step->GetTrack()->GetDefinition()->GetParticleName() == "gamma"   &&
       mIsDoseCorrectionTLEEnabled)) {
    mKFHandler->SetEnergy     (step->GetPreStepPoint()->GetKineticEnergy());
    mKFHandler->SetMaterial   (step->GetPreStepPoint()->GetMaterial());
    mKFHandler->SetDistance   (step->GetStepLength());
    mKFHandler->SetCubicVolume(GetDoselVolume());

    double dose(0.);
    double flux(0.);

    if (step->GetTrack()->GetDefinition()->GetParticleName() == "neutron") {
      dose = mKFHandler->GetDose();
      flux = mKFHandler->GetFlux();

      if (mIsDoseCorrectionEnabled)
        dose = mKFHandler->GetDoseCorrected();
    }
    else if (step->GetTrack()->GetDefinition()->GetParticleName() == "gamma" &&
             mIsDoseCorrectionTLEEnabled)
      dose = mKFHandler->GetDoseCorrectedTLE();

    bool sameEvent = true;

    if (mIsLastHitEventImageEnabled) {
      if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
        sameEvent = false;
        mLastHitEventImage.SetValue(index, mCurrentEvent);
      }
    }

    GateMessage("Actor", 2,  "GateNTLEDoseActor -- UserSteppingActionInVoxel:" << Gateendl
         << " Dosel index    = " << index << Gateendl
         << " Dosel volume   = " << G4BestUnit(GetDoselVolume(), "Volume") << Gateendl
         << " Dosel material = " << step->GetPreStepPoint()->GetMaterial()->GetName() << Gateendl
         << " Particle       = " << step->GetTrack()->GetDefinition()->GetParticleName() << Gateendl
         << " KinEnergy      = " << G4BestUnit(step->GetPreStepPoint()->GetKineticEnergy(), "Energy") << Gateendl
         << " Distance       = " << G4BestUnit(step->GetStepLength(), "Length") << Gateendl
         << " Dose           = " << G4BestUnit(dose, "Dose") << Gateendl
         << " Flux           = " << flux << " neutrons/m⁻²" << Gateendl);


    if (mIsDoseImageEnabled) {
      if (mIsDoseUncertaintyImageEnabled || mIsDoseSquaredImageEnabled) {
        if (sameEvent) mDoseImage.AddTempValue(index, dose);
        else mDoseImage.AddValueAndUpdate(index, dose);
      }
      else
        mDoseImage.AddValue(index, dose);
    }

    if (mIsFluxImageEnabled) {
      if (mIsFluxUncertaintyImageEnabled || mIsFluxSquaredImageEnabled) {
        if (sameEvent) mFluxImage.AddTempValue(index, flux);
        else mFluxImage.AddValueAndUpdate(index, flux);
      }
      else
        mFluxImage.AddValue(index, flux);
    }

    if (mIsKermaFactorDumped)
    {
      bool found(false);
      for(size_t i=0; i < mMaterialList.size(); i++)
        if (mMaterialList[i] == step->GetPreStepPoint()->GetMaterial()->GetName())
          found = true;

      if(!found)
      {
        mMaterialList.push_back(step->GetPreStepPoint()->GetMaterial()->GetName());
        mg->Add(mKFHandler->GetKermaFactorGraph());
      }
    }
  }
  else if (mIsKillSecondaryEnabled)
    step->GetTrack()->SetTrackStatus(fStopAndKill);
}
//-----------------------------------------------------------------------------
