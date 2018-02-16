/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class GateThermalActor
  \author vesna.cuplov@gmail.com
  \brief Class GateThermalActor : This actor produces voxelised images of the heat diffusion in tissue.

*/

#ifndef GATETHERMALACTOR_HH
#define GATETHERMALACTOR_HH

#include <G4NistManager.hh>
#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateThermalActorMessenger.hh"
#include "GateImageWithStatistic.hh"

#include "G4Event.hh"
#include <time.h>

// itk
#include "GateConfiguration.h"
#ifdef GATE_USE_ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageIterator.h>
#include "itkRecursiveGaussianImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkAddImageFilter.h"

#endif


class GateThermalActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateThermalActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateThermalActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void EndOfRunAction(const G4Run*); // default action (save)
  virtual void BeginOfEventAction(const G4Event * event);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track);
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  //  Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Scorer related
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  void setTime(G4double t);
  void setDiffusivity(G4double diffusivity);
  void setBloodPerfusionRate(G4double bloodperfusionrate);
  void setBloodDensity(G4double blooddensity);
  void setBloodHeatCapacity(G4double bloodheatcapacity);
  void setTissueDensity(G4double tissuedensity);
  void setTissueHeatCapacity(G4double tissueheatcapacity);
  void setScale(G4double simuscale);
  void setNumberOfTimeFrames(G4int numtimeframe);

protected:

  G4double mTimeNow;

  GateThermalActor(G4String name, G4int depth=0);
  GateThermalActorMessenger * pMessenger;

  int mCurrentEvent;
  int counter;

  StepHitType mUserStepHitType;

  GateImageWithStatistic mAbsorptionImage;

  G4String mAbsorptionFilename;
  G4String mHeatDiffusionFilename;

  G4double mUserDiffusionTime;
  G4double mUserMaterialDiffusivity;
  G4double mUserSimulationScale;

  G4int    mUserNumberOfTimeFrames;

  G4double deltaT;
  G4double mUserBloodPerfusionRate;
  G4double mUserBloodDensity;
  G4double mUserBloodHeatCapacity;
  G4double mUserTissueDensity;
  G4double mUserTissueHeatCapacity;

};

MAKE_AUTO_CREATOR_ACTOR(ThermalActor,GateThermalActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
