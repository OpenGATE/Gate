/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*!
  \class  GateTLFluenceActor
  \author anders.garpebring@umu.se
*/ 

#ifndef GATETLFLUENCEACTOR_HH
#define GATETLFLUENCEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateTLFluenceActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "G4UnitsTable.hh"

class GateTLFluenceActor : public GateVImageActor 
{
public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateTLFluenceActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateTLFluenceActor) 

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableFluenceImage(bool b) { isFluenceImageEnabled = b; }
  void EnableEnergyFluenceImage(bool b) { isEnergyFluenceImageEnabled = b; }
  
  void EnableFluenceSquaredImage(bool b) { isFluenceSquaredImageEnabled = b; }
  void EnableEnergyFluenceSquaredImage(bool b) { isEnergyFluenceSquaredImageEnabled = b; }
  void EnableFluenceUncertaintyImage(bool b) { isFluenceUncertainyImageEnabled = b; }
  void EnableEnergyFluenceUncertaintyImage(bool b) { isEnergyFluenceUncertaintyImageEnabled = b; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  //virtual void PostUserTrackingAction(const GateVVolume *, const G4Track* t);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void UserSteppingActionInVoxel(const int /*index*/, const G4Step* /*step*/){}
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateTLFluenceActor(G4String name, G4int depth=0);
  GateTLFluenceActorMessenger * pMessenger;

  GateImageWithStatistic fluenceImage; 
  GateImageWithStatistic energyFluenceImage;
  GateImageInt lastHitEventImage;
//  GateImage mLastHitEventImage; // Remove?

  G4String fluenceFilename;
  G4String energyFluenceFilename;
  G4String fluenceSquaredFilename;
  G4String energyFluenceSquaredFilename;
  
  G4double voxelVolume;
  std::vector<G4double> voxelDimensions;
  
  
  bool isFluenceImageEnabled;
  bool isEnergyFluenceImageEnabled;
  bool isFluenceSquaredImageEnabled;
  bool isEnergyFluenceSquaredImageEnabled;
  bool isFluenceUncertainyImageEnabled;
  bool isEnergyFluenceUncertaintyImageEnabled;
  bool isLastHitEventImageEnabled;

  int currentEvent;
  G4double nValuesPerVoxel;

  // Helper methods
private:
  void storeFluenceAtCurrentIndex(int index,G4double fluence, G4double energyFluence);
};

MAKE_AUTO_CREATOR_ACTOR(TLFluenceActor,GateTLFluenceActor)

#endif /* end #define GATETLFluenceACTOR_HH */
