/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class GateHybridMultiplicityActor
  \author francois.smekens@creatis.insa-lyon.fr
 */

#ifndef GATEHYBRIDMULTIPLICITYACTOR_HH
#define GATEHYBRIDMULTIPLICITYACTOR_HH

#include "GateVActor.hh"
#include "GatePhysicsList.hh"
#include "GateActorManager.hh"
#include "GateMaterialMuHandler.hh"
#include "GateHybridDoseActor.hh"
#include "GateHybridMultiplicityActorMessenger.hh"
#include "G4UnitsTable.hh"
#include "G4ParticleTable.hh"

//-----------------------------------------------------------------------------

class GateHybridMultiplicityActor : public GateVActor
{
 public: 
  
  virtual ~GateHybridMultiplicityActor();
    
  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateHybridMultiplicityActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();
  virtual void BeginOfEventAction(const G4Event *);
  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step *);  
  
  void SetPrimaryMultiplicity(int m) { defaultPrimaryMultiplicity = m; }
  void SetSecondaryMultiplicity(int m) { defaultSecondaryMultiplicity = m; }
  
  G4double GetHybridTrackWeight() { return currentHybridTrackWeight; }
  void SetHybridTrackWeight(G4double w) { currentHybridTrackWeight = w; }
  
  int AddSecondaryMultiplicity(G4VPhysicalVolume *);
  
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();
  
protected:

  GateHybridMultiplicityActor(G4String name, G4int depth=0);

  // secondary multiplicity can be different in each volume
  int defaultPrimaryMultiplicity;
  int defaultSecondaryMultiplicity;
  std::map<G4String,int> secondaryMultiplicityMap;
  
  G4double currentHybridTrackWeight;
  GateMaterialMuHandler* materialHandler;
  G4ProcessVector *processListForGamma;
      
  // store the track and the associated hybridWeight
  std::vector<G4Track *> theListOfHybridTrack;
  std::vector<G4double> theListOfHybridWeight;
  
  GateHybridMultiplicityActorMessenger *pActor;
};

MAKE_AUTO_CREATOR_ACTOR(HybridMultiplicityActor,GateHybridMultiplicityActor)


#endif /* end #define GATESHYBRIDMULTIPLICITYACTOR_HH */

