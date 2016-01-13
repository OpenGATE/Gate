/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class GateNanoActor
  \author vesna.cuplov@gmail.com
  \brief Class GateNanoActor : This actor produces a voxelised image of the energy deposited by optical
                               photons in the nano material (physics process NanoAbsorption): absorption map.
                               This absorption map corresponds to an initial condition (heat at t=0). The heat 
                               is then diffused and a second voxelised image is obtained as the solution of the 
                               heat equation at a later time.
*/


#ifndef GATENANOACTOR_HH
#define GATENANOACTOR_HH

#include <G4NistManager.hh>
#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateNanoActorMessenger.hh"
#include "GateImageWithStatistic.hh"

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
#endif


class GateNanoActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateNanoActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateNanoActor)

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
  void setBodyTemperature(G4double bodytemperature);
  void setBloodTemperature(G4double bloodtemperature);
  void setNanoMaximumTemperature(G4double nanotemperature);
  void setBloodPerfusionRate(G4double bloodperfusionrate);
  void setBloodDensity(G4double blooddensity);
  void setBloodHeatCapacity(G4double bloodheatcapacity);
  void setTissueDensity(G4double tissuedensity);
  void setTissueHeatCapacity(G4double tissueheatcapacity);
  void setTissueThermalConductivity(G4double tissuethermalconductivity);
  void setNanoAbsorptionCrossSection(G4double nanoabsorptionCS);
  void setNanoDensity(G4double nanodensity);
  void setScale(G4double simuscale);

protected:
  GateNanoActor(G4String name, G4int depth=0);
  GateNanoActorMessenger * pMessenger;

  int mCurrentEvent;
  StepHitType mUserStepHitType;

  GateImageWithStatistic mNanoAbsorptionImage;

  G4String mNanoAbsorptionFilename;
  G4String mNanoAbsorptioninTemperatureFilename;
  G4String mHeatConductionFilename;
  G4String mHeatConductionAdvectionFilename;

  G4double mUserDiffusionTime;
  G4double mUserMaterialDiffusivity;
  G4double mUserBodyTemperature;
  G4double mUserBloodTemperature;
  G4double mUserNanoTemperature;
  G4double mUserNanoAbsorptionCS;
  G4double mUserNanoDensity;
  G4double mUserSimulationScale;

  G4double deltaT;
  G4double mUserBloodPerfusionRate;
  G4double mUserBloodDensity;
  G4double mUserBloodHeatCapacity;
  G4double mUserTissueDensity;
  G4double mUserTissueHeatCapacity;
  G4double mUserTissueThermalConductivity;

};

MAKE_AUTO_CREATOR_ACTOR(NanoActor,GateNanoActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
