/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

//#include "GateConfiguration.h"

#ifndef GATEDOSESPECTRUMACTOR_HH
#define GATEDOSESPECTRUMACTOR_HH

#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateDoseSpectrumActorMessenger.hh"
#include "GateVActor.hh"


//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class G4EmCalculator;

class GateDoseSpectrumActor : public GateVActor
{
 public: 
    
  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  virtual ~GateDoseSpectrumActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateDoseSpectrumActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run * r);
  virtual void BeginOfEventAction(const G4Event *) ;
  virtual void UserSteppingAction(const GateVVolume *, const G4Step* step);

  //virtual void EndOfEventAction(const G4Event*);
  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();
  void DosePrimaryOnly(bool b){mDosePrimaryOnly = b;}

protected:
  GateDoseSpectrumActor(G4String name, G4int depth=0);

  bool mDosePrimaryOnly;
  double mVolumeMass;
  //double mEnergyDepot;
  int mLastHitEventImage;
  int mCurrentEvent;
  G4double mEventEnergy, mOldEventEnergy, mDoseEnergyTemp, mEnergyEventTemp;
  //double DOSIS;
  std::map< G4double, G4int > mNumParticPerEnergy;
  //std::map< G4int, G4double > mDoseEnergyTemp;
  //std::map< G4int, G4double > mEnergyEventTemp;
  std::map< G4double, G4double > mDoseEnergy;
  std::map< G4double, G4double > mDoseEnergySquare;
  GateDoseSpectrumActorMessenger* pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(DoseSpectrumActor,GateDoseSpectrumActor)


#endif /* end #define GATEDOSESPECTRUMACTOR_HH */

