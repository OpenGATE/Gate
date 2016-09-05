#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*!
  \class  GateAugerDetectorActor
  \author pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEAUGERDETECTORACTOR_HH
#define GATEAUGERDETECTORACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

#include <TFile.h>
#include <TH1.h>
#include <list>

struct AugerDeposition
{
  G4ThreeVector position;
  G4double energy;
  G4double time;
};

typedef std::list<AugerDeposition> AugerDepositions;

///----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateAugerDetectorActor : public GateVActor
{
public:

  virtual ~GateAugerDetectorActor();
  void setMinTOF(G4double tof);
  void setMaxTOF(G4double tof);
  void setMinEdep(G4double edep);
  void setMaxEdep(G4double edep);
  void setProjectionDirection(const G4ThreeVector& dir);
  void setMinimumProfileAxis(G4double min);
  void setMaximumProfileAxis(G4double max);
  void setProfileSize(int nbpts);
  void setProfileNoiseFWHM(G4double noise_fwhm);

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateAugerDetectorActor)

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

protected:
  GateAugerDetectorActor(G4String name, G4int depth=0);

  G4double GetTotalDepositedEnergy() const;
  G4ThreeVector GetWeighedBarycenterPosition() const;
  G4double GetWeighedBarycenterTime() const;

  TFile* pTfile;
  TH1D* pProfileHisto;
  TH1D* pEnergyDepositionHisto;
  TH1D* pTimeOfFlightHisto;

  GateActorMessenger * pMessenger;
  AugerDepositions depositions;
  G4double min_time_of_flight;
  G4double max_time_of_flight;
  G4double min_energy_deposition;
  G4double max_energy_deposition;
  G4ThreeVector projection_direction;
  G4double profile_min;
  G4double profile_max;
  int profile_nbpts;
  G4double profile_noise_fwhm;
};

MAKE_AUTO_CREATOR_ACTOR(AugerDetectorActor,GateAugerDetectorActor)


#endif /* end #define GATEAUGERDETECTORACTOR_HH */
#endif
