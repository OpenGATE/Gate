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
#include "GateDiscreteSpectrum.hh"

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

class G4EmCalculator;
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

  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  double GetEmin() {return mEmin; }
  double GetEmax() {return mEmax;}
  int GetENBins() {return mENBins;}
  
  G4double GetLETmin() {return mLETmin; }
  G4double GetLETmax() {return mLETmax; }
  int GetNLETBins() {return mLETBins; }
  void SetLETmin(double v) {mLETmin = v;}
  void SetLETmax(double v) {mLETmax = v;}
  void SetNLETBins(double v) {mLETBins = v;}

  void SetEmin(double v) {mEmin = v;}
  void SetEmax(double v) {mEmax = v;}
  void SetENBins(int v) {mENBins = v;}

  double GetEdepmin() {return mEdepmin; }
  double GetEdepmax() {return mEdepmax;}
  int GetEdepNBins() {return mEdepNBins;}

  void SetEdepmin(double v) {mEdepmin = v;}
  void SetEdepmax(double v) {mEdepmax = v;}
  void SetEdepNBins(int v) {mEdepNBins = v;}
  void SetLETSpectrumCalc(bool b) {mEnableLETSpectrumFlag = b; }
  void SetSaveAsTextFlag(bool b) { mSaveAsTextFlag = b; }
  void SetSaveAsTextDiscreteEnergySpectrumFlag(bool b) { mSaveAsDiscreteSpectrumTextFlag = b; if (b) SetSaveAsTextFlag(b); }

protected:
  GateEnergySpectrumActor(G4String name, G4int depth=0);

  TFile * pTfile;
  G4String mHistName;

  TH1D * pEnergySpectrum;
  TH1D * pDeltaEc;
  TH1D * pEdep;
  TH2D * pEdepTime;
  TH1D * pEdepTrack;

  TH1D * pLETSpectrum;
  G4double mLETmin;
  G4double mLETmax;
  int mLETBins;
  
  double mEmin;
  double mEmax;
  int mENBins;

  double mEdepmin;
  double mEdepmax;
  int mEdepNBins;

  double Ei,Ef;
  int nTrack;
  int nEvent;
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


  GateDiscreteSpectrum mDiscreteSpectrum;
  void SaveAsText(TH1D * histo, G4String initial_filename);
  bool mSaveAsTextFlag;
  bool mSaveAsDiscreteSpectrumTextFlag;
  bool mEnableLETSpectrumFlag;
  
  G4EmCalculator * emcalc;
};

MAKE_AUTO_CREATOR_ACTOR(EnergySpectrumActor,GateEnergySpectrumActor)


#endif /* end #define GATEENERGYSPECTRUMACTOR_HH */
#endif
