/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*!
  \class  GateEnergySpectrumActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
	  pierre.gueth@creatis.insa-lyon.fr
 */

#ifndef GATEENERGYSPECTRUMACTOR_HH
#define GATEENERGYSPECTRUMACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateEnergySpectrumActor : public GateVActor
{
 public:

  virtual ~GateEnergySpectrumActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateEnergySpectrumActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run * r);
  virtual void BeginOfEventAction(const G4Event *) ;
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) ;
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;
  virtual void EndOfEventAction(const G4Event*);
  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

//  virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  double GetEmin() {return mEmin; }
  double GetEmax() {return mEmax;}
  int GetENBins() {return mENBins;}

  void SetEmin(double v) {mEmin = v;}
  void SetEmax(double v) {mEmax = v;}
  void SetENBins(int v) {mENBins = v;}

  double GetEdepmin() {return mEdepmin; }
  double GetEdepmax() {return mEdepmax;}
  int GetEdepNBins() {return mEdepNBins;}

  void SetEdepmin(double v) {mEdepmin = v;}
  void SetEdepmax(double v) {mEdepmax = v;}
  void SetEdepNBins(int v) {mEdepNBins = v;}


protected:
  GateEnergySpectrumActor(G4String name, G4int depth=0);

  TFile * pTfile;
  G4String mHistName;

  TH1D * pEnergySpectrum;
  TH1D * pDeltaEc;
  TH1D * pEdep;
  TH2D * pEdepTime;
  TH1D * pEdepTrack;

  double mEmin;
  double mEmax;
  int mENBins;

  double mEdepmin;
  double mEdepmax;
  int mEdepNBins;

  double Ei,Ef;
  int nTrack;
  bool newEvt;
  bool newTrack;
  double sumNi;
  double sumM1;
  double sumM2;
  double sumM3;

  double edep;
  double tof;
  double edepTrack;

  GateActorMessenger* pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(EnergySpectrumActor,GateEnergySpectrumActor)


#endif /* end #define GATEENERGYSPECTRUMACTOR_HH */
#endif
