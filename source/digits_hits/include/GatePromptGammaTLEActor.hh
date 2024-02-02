/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH
#define GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH

#include "GateConfiguration.h"
#include "GateVImageActor.hh"
#include "GateActorMessenger.hh"
#include "GatePromptGammaTLEActorMessenger.hh"
#include "GateImageOfHistograms.hh"
#include "GatePromptGammaData.hh"
#include "GateVImageVolume.hh"

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
  //virtual void EndOfEventAction(const G4Event * e);

  void SetInputDataFilename(std::string filename);
  virtual void SaveData();
  virtual void ResetData();

  void SetTimeNbBins(int tbins) { data.SetTimeNbBins(tbins); } // nb of proton time bins
  
  void EnableDebugOutput(bool b) { mIsDebugOutputEnabled = b; }
  void EnableOutputMatch(bool b) { mIsOutputMatchEnabled = b; }
  //void EnableSysVarianceImage(bool b) { mIsSysVarianceImageEnabled = b; }
  //void EnableIntermediaryUncertaintyOutput(bool b) { mIsIntermediaryUncertaintyOutputEnabled = b; }

protected:
  GatePromptGammaTLEActor(G4String name, G4int depth=0);
  GatePromptGammaTLEActorMessenger * pMessenger;

  std::string mInputDataFilename;
  GatePromptGammaData data;
  bool alreadyHere;

  bool mIsDebugOutputEnabled;
  bool mIsOutputMatchEnabled;

  //helper functions
  void SetTrackIoH(GateImageOfHistograms*&);
  void SetTLEIoH(GateImageOfHistograms*&);
  void SetTofIoH(GateImageOfHistograms*&);
  GateVImageVolume* GetPhantom();
  void BuildVarianceOutput(); //converts trackl,tracklsq into mImageGamma and tlevar per voxel. Not used.
  //void BuildSysVarianceOutput(); //converts trackl into mImageGamma and tlesysvarv. Not used.

  //used and reset each track
  GateImageOfHistograms * tmptrackl;    //l_i

  //updated at end of event:
  GateImageOfHistograms * trackl;       //L_i. also intermediate output: track length per voxel per E_proton
  GateImageOfHistograms * tracklsq;     //L_i^2. also intermediate output: track squared length per voxel per E_proton

  //output, calculated at end of simu
  GateImageOfHistograms * mImagetof;
  GateImageOfHistograms * mImageGamma;  //oldstyle main output,
  GateImageOfHistograms * tle;          //main output (yield)
  GateImageOfHistograms * tlesysvar;    //systematic variance per voxel, per E_gamma. Not used.
  GateImageOfHistograms * tlevariance;  //uncertainty per voxel, per E_gamma. Not used.

  GateImageInt mLastHitEventImage;      //store eventID when last updated.
  int mCurrentEvent;                    //monitor event. TODO: not sure if necesary
  int mCurrentIndex;                    //monitor the input and output of the particule in a voxel /** Modif Oreste **/
  double startEvtTime;

  TH1D * pTime; // TH1D buffer histogram to add a new time value 
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaTLEActor,GatePromptGammaTLEActor)

#endif // end GATEPROMPTGAMMAPRODUCTIONTLEACTOR
