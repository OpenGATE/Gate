/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  Gate_NN_ARF_Actor
*/

#include "GateConfiguration.h"

#ifndef GATE_NN_ARF_ACTOR_HH
#define GATE_NN_ARF_ACTOR_HH

#include "GateActorManager.hh"
#include "GateMiscFunctions.hh"
#include "GateVActor.hh"
#include "Gate_NN_ARF_ActorMessenger.hh"
#include "GateImage.hh"

#ifdef GATE_USE_TORCH
#include <torch/script.h>
#endif

//-----------------------------------------------------------------------------
struct Gate_NN_ARF_Train_Data {
  double theta; // in deg, angle along X
  double phi;   // in deg, angle along Y
  double E;     // in MeV
  double w;     // windows id (0 if outside)
  // Helper
  void Print(std::ostream & os);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
struct Gate_NN_ARF_Test_Data {
  double x;     // in mm
  double y;     // in mm
  double theta; // in deg, angle along X
  double phi;   // in deg, angle along Y
  double E;     // in MeV
  std::vector<double> nn; // output of the neural network
  // Helper
  void Print(std::ostream & os);
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
class Gate_NN_ARF_Actor: public GateVActor
{
public:

  // Macro to auto declare actor
  FCT_FOR_AUTO_CREATOR_ACTOR(Gate_NN_ARF_Actor)

  // Actor name
  virtual ~Gate_NN_ARF_Actor();

  // Constructs the sensor
  virtual void Construct();

  // Parameters
  void SetEnergyWindowNames(std::string & names);
  void SetMode(std::string m);
  void SetMaxAngle(double a);
  void SetRRFactor(int f);
  void SetNNModel(std::string& m);
  void SetNNDict(std::string& m);
  void SetImage(std::string& m);
  void SetSpacing(double m, int index);
  void SetSize(int m, int index);
  void SetCollimatorLength(double m);
  void SetScale(double m);
  void SetBatchSize(double m);

  // Callbacks
  virtual void BeginOfRunAction(const G4Run *);
  virtual void BeginOfEventAction(const G4Event * e);
  virtual void EndOfEventAction(const G4Event * e);
  virtual void UserSteppingAction(const GateVVolume * v, const G4Step* step);

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Apply NN to current batch of particles
  void ProcessBatch();

protected:
  Gate_NN_ARF_Actor(G4String name, G4int depth = 0);
  Gate_NN_ARF_ActorMessenger * pMessenger;

  bool mTrainingModeFlag;
  bool mEnergyModeFlag;
  bool mIgnoreCurrentData;
  std::vector<Gate_NN_ARF_Test_Data>  mTestData;
  std::vector<Gate_NN_ARF_Train_Data> mTrainData;
  bool mEventIsAlreadyStored;
  Gate_NN_ARF_Test_Data mCurrentTestData;
  Gate_NN_ARF_Train_Data mCurrentTrainData;
  GateImageDouble* mImage;
  std::vector<G4String> mListOfWindowNames;
  std::vector<int> mListOfWindowIds;
  int mNumberOfDetectedEvent;
  int mRRFactor;
  double mMaxAngle;
  double mThetaMax;
  double mPhiMax;
  std::vector<double> mSpacing; //Spacing in mm of the image
  std::vector<int> mSize; //Size in pixel of the image
  double mCollimatorLength; //collimator+ half crystal length in mm
  int mNDataset;
  int mNumberOfBatch;
  std::string mNNModelPath;
  std::string mNNDictPath;
  std::string mImagePath;
  std::vector<double> mXmean;
  std::vector<double> mXstd;
#ifdef GATE_USE_TORCH
  std::shared_ptr<torch::jit::script::Module> mNNModule;
  at::Tensor mNNOutput;
#endif
  float mBatchSize; //not unsigned int to be able to be superior to max int
  std::vector<std::vector<double> > mBatchInputs;
  unsigned int mCurrentSaveNNOutput;
};

// Macro to auto declare actor
MAKE_AUTO_CREATOR_ACTOR(NN_ARF_Actor, Gate_NN_ARF_Actor)

#endif /* end #define GATEDETECTORINOUT_HH */
