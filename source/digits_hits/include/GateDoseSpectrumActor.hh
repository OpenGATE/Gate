/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*!
  \class  GateDoseSpectrumActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
	  pierre.gueth@creatis.insa-lyon.fr
 */

#ifndef GATEDOSESPECTRUMACTOR_HH
#define GATEDOSESPECTRUMACTOR_HH

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

//-----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateDoseSpectrumActor : public GateVActor
{
 public: 
  
  virtual ~GateDoseSpectrumActor();
    
  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateDoseSpectrumActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks

  virtual void BeginOfRunAction(const G4Run * r);
  virtual void BeginOfEventAction(const G4Event *) ;
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void EndOfEventAction(const G4Event*);
  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();
  void DosePrimaryOnly(bool b){mDosePrimaryOnly = b;}

protected:
  GateDoseSpectrumActor(G4String name, G4int depth=0);

  bool mDosePrimaryOnly;
  double mVolumeMass;
  double mEnergyDepot;
  double mEventEnergy;
  double DOSIS;
  std::map< G4double, G4double> mDoseSpectrum;
  std::map< G4int, G4bool> mEnergyFlag;
  GateActorMessenger* pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(DoseSpectrumActor,GateDoseSpectrumActor)


#endif /* end #define GATEDOSESPECTRUMACTOR_HH */
#endif 
