/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateDoseActor
  \author claire.vanngocty@gmail.com
 */
#ifndef GATECROSSSECTIONPRODUCTIONACTOR_HH
#define GATECROSSSECTIONPRODUCTIONACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateCrossSectionProductionActorMessenger.hh"
#include "GateImageWithStatistic.hh"

#include "G4Event.hh"
#include <time.h>


class GateCrossSectionProductionActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateCrossSectionProductionActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateCrossSectionProductionActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

 /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related

  virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  void SetFilename(G4String f) { mIsotopeFilename = f; }
  void ActiveC11(bool b){m_IsC11=b; }

  void ActiveO15(bool b){m_IsO15=b; }
//new stuff 10/11/11
  virtual void EndOfEventAction(const G4Event* eve);
  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track* t);
  float InterpolLin (float xa , float ya, float xb, float yb, float X);
  double GetSectionEfficace(double nrj,std::map<float, float>& MapSigma);

protected:
  GateCrossSectionProductionActor(G4String name, G4int depth=0);
  GateCrossSectionProductionActorMessenger * pMessenger;
  std::map <float, float> SectionTableC11_C12;
  std::map <float, float> SectionTableC11_O16;
  std::map <float, float> SectionTableO15_O16;
  int mCurrentEvent;
  G4int nb_elemt_C12_in_table;
  G4int nb_elemt_O16_in_table;
  bool newTrack;

  std::map <int, int>PixelValuePerEvent;
  std::map <int, int>PixelValuePerEvent_secondary;
  //for C12
  GateImageWithStatistic *  mIsotopeImage;
  GateImage mEnergyImage;
  GateImage mStatImage;
  GateImage mEnergyImage_secondary;
  GateImage mStatImage_secondary;
  GateImage mDensityImage;
  GateImage mfractionC12Image;
  GateImage mfractionO16Image;

  G4String mIsotopeFilename;
  G4String mEnergyFilename;
  G4String mStatFilename;
  G4double threshold_energy_C12;

  G4double Na;
  G4double A_12;
  G4double A_16;
  G4double max_energy_cross_section;


  bool m_IsC11;
  bool m_IsO15;
  //O15
  GateImageWithStatistic * mIsotopeImage_O15;
  G4double threshold_energy_O16;
};

MAKE_AUTO_CREATOR_ACTOR(CrossSectionProductionActor,GateCrossSectionProductionActor)

#endif
