/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class GateTrackLengthActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#ifndef GATETRACKLENGTHACTOR_HH
#define GATETRACKLENGTHACTOR_HH

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateVActor.hh"
#include "GateTrackLengthActorMessenger.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"

//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateTrackLengthActor : public GateVActor
{
 public:

  virtual ~GateTrackLengthActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateTrackLengthActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event *) {}
  virtual void UserPreTrackActionInVoxel(const int, const G4Track*) {}
  virtual void UserPostTrackActionInVoxel(const int, const G4Track*) {}
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;


  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  double GetLmin() {return mLmin; }
  double GetLmax() {return mLmax;}
  int GetNBins() {return mNBins;}

  void SetLmin(double v) {mLmin = v;}
  void SetLmax(double v) {mLmax = v;}
  void SetNBins(int v) {mNBins = v;}


protected:
  GateTrackLengthActor(G4String name, G4int depth=0);

  TFile * pTfile;
  G4String mHistName;

  TH1D * pTrackLength;

  double mLmin;
  double mLmax;
  int mNBins;

  GateTrackLengthActorMessenger * pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(TrackLengthActor,GateTrackLengthActor)


#endif /* end #define GATETRACKLENGTHACTOR_HH */
#endif
