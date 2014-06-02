/*!
 */

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATEFRAGMENTATIONANDPRODUCTIONACTOR_HH
#define GATEFRAGMENTATIONANDPRODUCTIONACTOR_HH

#include "GateVActor.hh"

#include "GateActorManager.hh"

#include "GateFragmentationAndProductionActorMessenger.hh"
#include "GateVImageActor.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TVector2.h"

///----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateFragmentationAndProductionActor : public GateVActor
{
 public:

  virtual ~GateFragmentationAndProductionActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateFragmentationAndProductionActor)

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
  //virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  void SetNBins(unsigned int aNBins) { pNBins = aNBins; }
  unsigned int GetNBins() const { return pNBins; }

protected:
  GateFragmentationAndProductionActor(G4String name, G4int depth=0);

  unsigned int pNBins;
  TFile * pTFile;
  TH1D * pGammaProduction;
  TH1D * pNeutronProduction;
  TH1D * pFragmentation;
  TVector2 * pNEvent;

  GateFragmentationAndProductionActorMessenger * pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(FragmentationAndProductionActor,GateFragmentationAndProductionActor)


#endif /* end #define GATEFRAGMENTATIONANDPRODUCTIONACTOR_HH */
#endif
