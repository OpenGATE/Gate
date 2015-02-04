/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONANALOGACTOR_HH
#define GATEPROMPTGAMMAPRODUCTIONANALOGACTOR_HH

#include "GateConfiguration.h"
#include "GateVImageActor.hh"
#include "GateActorMessenger.hh"
#include "GatePromptGammaAnalogActorMessenger.hh"
#include "GateImageOfHistograms.hh"
#include "GatePromptGammaData.hh"

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaAnalogActor: public GateVImageActor
{
public:
  virtual ~GatePromptGammaAnalogActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaAnalogActor)

  virtual void Construct();
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void BeginOfEventAction(const G4Event * e);
  virtual void EndOfEventAction(const G4Event * e);

  void SetInputDataFilename(std::string filename);
  virtual void SaveData();
  virtual void ResetData();

  void EnableUncertaintyImage(bool b) { mIsIntermediaryUncertaintyOutputEnabled = mIsUncertaintyImageEnabled = b; }  //all is needed to calc tle uncertainty
  void EnableIntermediaryUncertaintyOutput(bool b) { mIsIntermediaryUncertaintyOutputEnabled = b; }

protected:
  GatePromptGammaAnalogActor(G4String name, G4int depth=0);
  GatePromptGammaAnalogActorMessenger * pMessenger;

  //we'll not use it, but extract the bins and binsizes for the PG output.
  std::string mInputDataFilename;
  GatePromptGammaData data;

  bool mIsUncertaintyImageEnabled;
  bool mIsIntermediaryUncertaintyOutputEnabled;

  int gammabin(double energy);
  bool mIsFistStep;
  TH1D * converterHist;          //sole use is to aid conversion of proton energy to bin index.

  GateImageOfHistograms * mImageGamma;  //main output (yield)

  //not sure if necesary
  GateImageInt mLastHitEventImage;      //store eventID when last updated. TODO: not sure if necesary
  int mCurrentEvent;                    //monitor event. TODO: not sure if necesary
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaAnalogActor,GatePromptGammaAnalogActor)

#endif // end GATEPROMPTGAMMAPRODUCTIONANALOGACTOR
