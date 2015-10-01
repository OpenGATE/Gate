/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateNanoActor
  \author 

  \date	July 2015
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
#include "itkRecursiveGaussianImageFilter.h"
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

  void EnableNanoAbsorptionImage(bool b) { mIsNanoAbsorptionImageEnabled = b; }

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


  void setGaussianSigma(G4double sigma);
  void setTime(G4double t);
  void setDiffusivity(G4double diffusivity);


protected:
  GateNanoActor(G4String name, G4int depth=0);
  GateNanoActorMessenger * pMessenger;

  int mCurrentEvent;
  StepHitType mUserStepHitType;

  bool mIsNanoAbsorptionImageEnabled;

  GateImageWithStatistic mNanoAbsorptionImage;

  G4String mNanoAbsorptionFilename;
  G4String mHeatDiffusionFilename;

  G4double gaussian_sigma;
  G4double diffusion_time;
  G4double material_diffusivity;


  G4double x;
  G4double y;
  G4double z;



};

MAKE_AUTO_CREATOR_ACTOR(NanoActor,GateNanoActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
