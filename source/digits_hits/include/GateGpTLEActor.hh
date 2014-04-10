#ifndef GATEGPTLEACTOR_HH
#define GATEGPTLEACTOR_HH

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TObject.h>

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

class GateGpTLEActor : public GateVActor
{
public: 
  virtual ~GateGpTLEActor();
  
  FCT_FOR_AUTO_CREATOR_ACTOR(GateGpTLEActor)
  
  virtual void Construct();
  
  virtual void BeginOfRunAction(const G4Run*);
  virtual void BeginOfEventAction(const G4Event*) ;
  virtual void UserSteppingAction(const GateVVolume*, const G4Step*);
  
  virtual void PreUserTrackingAction(const GateVVolume*, const G4Track*);
  virtual void PostUserTrackingAction(const GateVVolume*, const G4Track*);
  virtual void EndOfEventAction(const G4Event*);
  
  virtual void SaveData();
  virtual void ResetData();
  
  void SaveFilename(G4String name) { mSaveFilename = name; }
  
  void FileSpectreBaseName(G4String);
  void constructSpectrumMaterial(G4String);
  std::map< G4String, TH2D*> MaterialMap;

protected:
  GateGpTLEActor(G4String name, G4int depth=0);
  
  G4String mUserFileSpectreBaseName;
  G4String mSaveFilename;
  
  size_t last_secondaries_size;
  bool first_step;
  GateActorMessenger* pMessenger;
  TFile* pFile;
  TH1D* pHEgTLE;
  TH2D* pHEpEgpNormalized;
  TH1D* pHEpInelastic;
  TH1D* pHEpInelasticProducedGamma;
  TH1D* H_SUM;
  TH1D* Ep;
  G4double minX;
  G4double maxX;
  G4double minY;
  G4double maxY;
  G4int NbinsX;
  G4int NbinsY;
  
};

MAKE_AUTO_CREATOR_ACTOR(GpTLEActor,GateGpTLEActor)

#endif
#endif 
