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
  andreas.resch@meduniwien.ac.at
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
#include <TMath.h>

#include <list>

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
  
  G4double GetQmin() {return mQmin; }
  G4double GetQmax() {return mQmax; }
  int GetNQBins() {return mQBins; }
  void SetQmin(double v) {mQmin = v;}
  void SetQmax(double v) {mQmax = v;}
  void SetNQBins(double v) {mQBins = v;}

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
  void SetLETFluenceSpectrumCalc(bool b) {mEnableLETFluenceSpectrumFlag = b; }
  void SetLETtoMaterialFluenceSpectrumCalc(bool b) {mEnableLETtoMaterialFluenceSpectrumFlag = b; }
  void SetOtherMaterial(G4String b, bool c) { mOtherMaterial = b; mOtherMaterialFlag = c;}
  
  void SetQSpectrumCalc(bool b) {mEnableQSpectrumFlag = b; }
  void SetSaveAsTextFlag(bool b) { mSaveAsTextFlag = b; }
  void SetSaveAsTextDiscreteEnergySpectrumFlag(bool b) { mSaveAsDiscreteSpectrumTextFlag = b; if (b) SetSaveAsTextFlag(b); }

  void SetESpectrumNbPartCalc(bool b) {mEnableEnergySpectrumNbPartFlag = b; }
  void SetESpectrumFluenceCosCalc(bool b) {mEnableEnergySpectrumFluenceCosFlag = b; }
  void SetESpectrumFluenceTrackCalc(bool b) {mEnableEnergySpectrumFluenceTrackFlag = b; }
  void SetESpectrumEdepCalc(bool b) {mEnableEnergySpectrumEdepFlag = b; }

  void SetEdepHistoCalc(bool b) {mEnableEdepHistoFlag = b; }
  void SetEdepTimeHistoCalc(bool b) {mEnableEdepTimeHistoFlag= b; }
  void SetEdepTrackHistoCalc(bool b) {mEnableEdepTrackHistoFlag = b; }
  void SetElossHistoCalc(bool b) {mEnableElossHistoFlag = b; }
  
  void SetEdepStepHistoCalc(bool b) {mEnableEdepStepHistoFlag = b; }
  
  void SetLogBinning(bool b) {mEnableLogBinning = b; }
  void SetEnergyPerUnitMass(bool b) {mEnableEnergyPerUnitMass = b; }
  void SetRelativePrimEvents(bool b) {mEnableRelativePrimEvents = b; }
  
  TH1D* FactoryTH1D(const char *name, const char *title, Int_t nbinsx, Double_t xlow, Double_t xup, const char *xtitle, const char *ytitle);
  TH1D* FactoryTH1D2(const char *name, const char *title, const char *xtitle, const char *ytitle,  double* binV, int nbins);
 
  double* CreateBinVector(double emin, double emax, int nbins, bool enableLogBin);
  
protected:
  GateEnergySpectrumActor(G4String name, G4int depth=0);

  TFile * pTfile;
  G4String mHistName;

  TH1D * pEnergySpectrum;
  TH1D * pEnergySpectrumFluence;
  TH1D * pEnergySpectrumTrack;
  //TH1D * ;
  //TH1D * ;
  //TH1D * ;
  TH1D * pEnergySpectrumNbPart;
  TH1D * pEnergySpectrumFluenceCos;
  TH1D * pEnergySpectrumFluenceTrack;
  TH1D * pEnergySpectrumLET;
  TH1D * pEnergySpectrumLETdoseWeighted;
  
  TH1D * pEnergyEdepSpectrum;
  TH1D * pDeltaEc;
  TH1D * pEdep;
  TH2D * pEdepTime;
  TH1D * pEdepTrack;
  TH1D * pEdepStep;
  
  std::list<TH1D*> allEnabledTH1DHistograms;

  TH1D * pLETSpectrum;
  TH1D * pLETFluenceSpectrum;
  TH1D * pLETtoMaterialFluenceSpectrum;
  G4double mLETmin;
  G4double mLETmax;
  int mLETBins;
  G4double pEnergySpectrumTrackNorm;

  TH1D * pQSpectrum;
  G4double mQmin;
  G4double mQmax;
  int mQBins;
  
  double * eBinV;
  //double dEn;
  
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

  G4double edep;
  G4Material* OtherMaterial;
  double tof;
  G4double edepTrack;
  G4double edepEvent;

  GateActorMessenger* pMessenger;


  GateDiscreteSpectrum mDiscreteSpectrum;
  void SaveAsText(TH1D * histo, G4String initial_filename);
  bool mSaveAsTextFlag;
  bool mSaveAsDiscreteSpectrumTextFlag;
  bool mEnableLETSpectrumFlag;
  bool mEnableLETFluenceSpectrumFlag;
  bool mEnableLETtoMaterialFluenceSpectrumFlag;
  bool mEnableQSpectrumFlag;
  bool mEnableEnergySpectrumNbPartFlag;
  bool mEnableEnergySpectrumFluenceCosFlag;
  bool mEnableEnergySpectrumFluenceTrackFlag;
  bool mEnableEnergySpectrumEdepFlag;
  bool mEnableEdepStepHistoFlag;
  bool mEnableEdepHistoFlag;
  bool mEnableEdepTimeHistoFlag;
  bool mEnableEdepTrackHistoFlag;
  bool mEnableElossHistoFlag;
  bool mEnableLogBinning;
  bool mEnableEnergyPerUnitMass;
  bool mEnableRelativePrimEvents;
  bool mOtherMaterialFlag;
  G4String mOtherMaterial;
  
  G4EmCalculator * emcalc;

};



MAKE_AUTO_CREATOR_ACTOR(EnergySpectrumActor,GateEnergySpectrumActor)


#endif /* end #define GATEENERGYSPECTRUMACTOR_HH */
#endif
