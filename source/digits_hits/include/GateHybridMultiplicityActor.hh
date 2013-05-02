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
#include "G4UnitsTable.hh"
#include "G4ParticleTable.hh"
#include "G4VPhysicalVolume.hh"

//-----------------------------------------------------------------------------

class GateHybridMultiplicityActor : public GateVActor
{
public: 
   
  static GateHybridMultiplicityActor *GetInstance()
  {   
    if (singleton_HybridMultiplicityActor == 0)
    {
      //std::cout << "creating GateActorManager..." << std::endl;
      singleton_HybridMultiplicityActor = new GateHybridMultiplicityActor("hybridMultiplicityActor",0);
    }
    //else std::cout << "GateActorManager already created!" << std::endl;
    return singleton_HybridMultiplicityActor;
  };
  
  ~GateHybridMultiplicityActor();
    
  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateHybridMultiplicityActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  void Construct();
  void BeginOfEventAction(const G4Event *);
  void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  void PostUserTrackingAction(const GateVVolume *, const G4Track*);
  void UserSteppingAction(const GateVVolume *, const G4Step *);  
  
  void SetPrimaryMultiplicity(int m) { defaultPrimaryMultiplicity = m; }
  void SetSecondaryMultiplicity(int m) { defaultSecondaryMultiplicity = m; }
  int GetPrimaryMultiplicity() { return defaultPrimaryMultiplicity; }
  int GetSecondaryMultiplicity() { return defaultSecondaryMultiplicity; }  
  
  G4double GetHybridTrackWeight() { return currentHybridTrackWeight; }
  void SetHybridTrackWeight(G4double w) { currentHybridTrackWeight = w; };
  
  void SetMultiplicity(int, int, G4VPhysicalVolume *);
  
  /// Saves the data collected to the file
  void SaveData();
  void ResetData();
  
protected:

  GateHybridMultiplicityActor(G4String name, G4int depth=0);

  // secondary multiplicity can be different in each volume
  int defaultPrimaryMultiplicity;
  int defaultSecondaryMultiplicity;
  std::map<G4VPhysicalVolume *,int> secondaryMultiplicityMap;
  
  GateMaterialMuHandler* materialHandler;
  G4ProcessVector *processListForGamma;
      
  // store the track and the associated hybridWeight
  int currentTrackIndex;
  G4double currentHybridTrackWeight;
  std::vector<G4Track *> theListOfHybridTrack;
  std::vector<G4double> theListOfHybridWeight;
  
private:
  GateHybridMultiplicityActor();
  static GateHybridMultiplicityActor *singleton_HybridMultiplicityActor;
};

MAKE_AUTO_CREATOR_ACTOR(HybridMultiplicityActor,GateHybridMultiplicityActor)


#endif /* end #define GATESHYBRIDMULTIPLICITYACTOR_HH */

