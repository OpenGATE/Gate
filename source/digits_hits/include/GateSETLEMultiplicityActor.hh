/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class GateSETLEMultiplicityActor
  \author francois.smekens@creatis.insa-lyon.fr
 */

#ifndef GATESETLEMULTIPLICITYACTOR_HH
#define GATESETLEMULTIPLICITYACTOR_HH

#include "GateVActor.hh"
#include "GatePhysicsList.hh"
#include "GateActorManager.hh"
#include "GateMaterialMuHandler.hh"
#include "G4UnitsTable.hh"
#include "G4ParticleTable.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Hybridino.hh"

//-----------------------------------------------------------------------------

struct RaycastingStruct
{
  bool isPrimary;
  double energy;
  double weight;
  
  G4ThreeVector position;
  G4ThreeVector momentum;

  RaycastingStruct(bool b, double e, double w, G4ThreeVector p, G4ThreeVector m)
  {
    isPrimary = b;
    energy = e;
    weight = w;
    position = p;
    momentum = m;
  }
};

//-----------------------------------------------------------------------------

class GateSETLEMultiplicityActor : public GateVActor
{
public: 
   
  static GateSETLEMultiplicityActor *GetInstance()
  {   
    if (singleton_SETLEMultiplicityActor == 0)
    {
      //std::cout << "creating GateActorManager...\n";
      singleton_SETLEMultiplicityActor = new GateSETLEMultiplicityActor("seTLEMultiplicityActor",0);
    }
    //else std::cout << "GateActorManager already created!\n";
    return singleton_SETLEMultiplicityActor;
  };
  
  ~GateSETLEMultiplicityActor();
    
  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateSETLEMultiplicityActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  void Construct();
  void BeginOfEventAction(const G4Event *);
  void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  void PostUserTrackingAction(const GateVVolume *, const G4Track*);
  void UserSteppingAction(const GateVVolume *, const G4Step *);  
  
  void SetPrimaryMultiplicity(int m) { mDefaultPrimaryMultiplicity = m; }
  void SetSecondaryMultiplicity(int m) { mDefaultSecondaryMultiplicity = m; }
  int GetPrimaryMultiplicity() { return mDefaultPrimaryMultiplicity; }
  int GetSecondaryMultiplicity() { return mDefaultSecondaryMultiplicity; }  
  
  G4double GetHybridTrackWeight() { return mCurrentHybridTrackWeight; }
  void SetHybridTrackWeight(G4double w) { mCurrentHybridTrackWeight = w; };
  
  std::vector<RaycastingStruct> *GetRaycastingList() { return &mListOfRaycasting; }
  
  void SetMultiplicity(bool, int, int, G4VPhysicalVolume *);
  
  /// Saves the data collected to the file
  void SaveData();
  void ResetData();
  
protected:

  GateSETLEMultiplicityActor(G4String name, G4int depth=0);

  // secondary multiplicity can be different in each volume
  bool mIsHybridinoEnabled;
  int mDefaultPrimaryMultiplicity;
  int mDefaultSecondaryMultiplicity;
  std::map<G4VPhysicalVolume *,int> mSecondaryMultiplicityMap;
  G4ParticleDefinition *mHybridino;
  
  GateMaterialMuHandler* mMaterialHandler;
  G4ProcessVector *mProcessListForGamma;
      
  // store the track and the associated hybridWeight
  // - with hybridino
  int mCurrentTrackIndex;
  G4double mCurrentHybridTrackWeight;  
  std::vector<G4Track *> mListOfHybridTrack;
  std::vector<G4double> mListOfHybridWeight;
  // - without hybridino
  std::vector<RaycastingStruct> mListOfRaycasting;

private:
  GateSETLEMultiplicityActor();
  static GateSETLEMultiplicityActor *singleton_SETLEMultiplicityActor;
};

MAKE_AUTO_CREATOR_ACTOR(SETLEMultiplicityActor,GateSETLEMultiplicityActor)


#endif /* end #define GATESHYBRIDMULTIPLICITYACTOR_HH */

