/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class GateTLFluenceDistributionActor
  \author anders.garpebring@gmail.com
 */

#ifndef GATETLFLUENCEDISTRIBUTIONACTOR_HH
#define GATETLFLUENCEDISTRIBUTIONACTOR_HH

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateVActor.hh"
#include "GateTLFluenceDistributionActorMessenger.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"

#include <iostream>
#include <fstream>

//-----------------------------------------------------------------------------
/// \brief Actor storing energy and angular information about fluence.
class GateTLFluenceDistributionActor : public GateVActor
{
 public:

  virtual ~GateTLFluenceDistributionActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateTLFluenceDistributionActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);


  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}
  
  void enableEnergy(bool e){mEnergyEnabled = e;}
  void enableTheta(bool e){mThetaEnabled = e;}
  void enablePhi(bool e){mPhiEnabled = e;}
  
  void SetEnergyMin(double v) {mEnergyMin = v;}
  void SetEnergyMax(double v) {mEnergyMax = v;}
  void SetNEnergyBins(int v) {mEnergyBins = v;}

  void SetThetaMin(double v) {mThetaMin = v;}
  void SetThetaMax(double v) {mThetaMax = v;}
  void SetNThetaBins(int v) {mThetaBins = v;}
  
  void SetPhiMin(double v) {mPhiMin = v;}
  void SetPhiMax(double v) {mPhiMax = v;}
  void SetNPhiBins(int v) {mPhiBins = v;}
  
  void SetAsciiFile(G4String str) {mAsciiFileEnabled = true; mAsciiFileName = str;}

protected:
  GateTLFluenceDistributionActor(G4String name, G4int depth=0);

  TFile * pTfile;
  std::ofstream mAsciiFile;
  
  G4String mHistName;
  G4String mAsciiFileName;
  
  bool mEnergyEnabled;
  bool mThetaEnabled;
  bool mPhiEnabled;
  bool mAsciiFileEnabled;
  
  G4double mEnergyMax, mEnergyMin, mThetaMin, mThetaMax, mPhiMin, mPhiMax;
  
  int mEnergyBins, mThetaBins, mPhiBins;

  TH1D *pHistEnergy, *pHistTheta, *pHistPhi;
  TH2D *pHistEnergyTheta, *pHistEnergyPhi, *pHistThetaPhi;
  
  G4double detectorVolume;

  GateTLFluenceDistributionActorMessenger * pMessenger;
  
  // Helper functions
  G4ThreeVector GetLocalDirection(const G4Step *step);
  void storeStepInHistograms(G4double energy, G4double theta, G4double phi, G4double dF);
};

MAKE_AUTO_CREATOR_ACTOR(TLFluenceDistributionActor,GateTLFluenceDistributionActor)


#endif /* end #define GATETLFLUENCEDISTRIBUTIONACTOR_HH */
#endif
