/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH
#define GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH

#include "GateConfiguration.h"
#include "GateVImageActor.hh"
#include "GateActorMessenger.hh"
#include "GatePromptGammaTLEActorMessenger.hh"
#include "GateImageOfHistograms.hh"
#include "GatePromptGammaData.hh"
#include "GateImageWithStatisticTLE.hh"

#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaTLEActor: public GateVImageActor
{
public:
  virtual ~GatePromptGammaTLEActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaTLEActor)

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
  GatePromptGammaTLEActor(G4String name, G4int depth=0);
  GatePromptGammaTLEActorMessenger * pMessenger;

  std::string mInputDataFilename;
  GatePromptGammaData data;

  bool mIsUncertaintyImageEnabled;
  bool mIsIntermediaryUncertaintyOutputEnabled;

  //used and reset each track
  GateImageOfHistograms * tmptrackl;    //l_i
  int protbin(double energy);
  TH1D * converterHist;          //sole use is to aid conversion of proton energy to bin index.

  //updated at end of track:
  GateImageOfHistograms * trackl;       //L_i. also intermediate output: track length per voxel per E_proton
  GateImageOfHistograms * tracklsq;     //L_i^2. also intermediate output: track squared length per voxel per E_proton
  GateImageOfHistograms * tleuncertain; //uncertainty per voxel, per E_gamma

  //FIXME: allocate and calculate at end of simulation
  GateImageOfHistograms * mImageGamma;  //main output (yield)

  //written at end of simu in case of uncertainty output
  //GateImageOfHistograms * gammam;     //intermediate output: Gamma_m database, per E_proton per E_gamma

  //not sure if necesary
  GateImageInt mLastHitEventImage;      //store eventID when last updated. TODO: not sure if necesary
  int mCurrentEvent;                    //monitor event. TODO: not sure if necesary
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaTLEActor,GatePromptGammaTLEActor)

#endif // end GATEPROMPTGAMMAPRODUCTIONTLEACTOR
