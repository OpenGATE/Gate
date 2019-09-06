/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "json.hpp"
#include "Gate_NN_ARF_Actor.hh"
#include "GateSingleDigi.hh"
#include "G4DigiManager.hh"
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TBranch.h"
#include "TString.h"
#include <iostream>
//-----------------------------------------------------------------------------
void Gate_NN_ARF_Train_Data::Print(std::ostream & os)
{
  os << " train = "
     << theta << " "
     << phi << " "
     << E << " : "
     << w << " "
     << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void Gate_NN_ARF_Test_Data::Print(std::ostream & os)
{
  os << " test = "
     << x << " "
     << y << " "
     << theta << " "
     << phi << " "
     << E << " "
     << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::Gate_NN_ARF_Actor(G4String name, G4int depth) :
  GateVActor(name, depth)
{
  GateDebugMessageInc("Actor",4,"Gate_NN_ARF_Actor() -- begin\n");
  pMessenger = new Gate_NN_ARF_ActorMessenger(this);
  mTrainingModeFlag = false;
  mEnergyModeFlag = false;
  mMaxAngle = 0.0; // no max angle
  mRRFactor = 0;   // no Russian Roulette factor
  mThetaMax = 0.0;
  mPhiMax = 0.0;
  mSize.resize(2);
  mSpacing.resize(2);
  mCollimatorLength = 99;
  mNDataset = 0;
  mBatchSize = 1e5;
  mCurrentSaveNNOutput = 0;
  GateDebugMessageDec("Actor",4,"Gate_NN_ARF_Actor() -- end\n");
  mNNModelPath = "";
  mNNDictPath = "";
  mNumberOfBatch = 0;
  mImage = new GateImageDouble();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::~Gate_NN_ARF_Actor()
{
  delete pMessenger;
  delete mImage;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetEnergyWindowNames(std::string & names)
{
  std::vector<std::string> words;
  GetWords(words, names);
  for(auto w:words) mListOfWindowNames.push_back(w);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetMode(std::string m)
{
  bool found = false;
  if (m == "train") {
    mTrainingModeFlag = true;
    mEnergyModeFlag = false;
    found = true;
  }
  if (m == "test") {
    mTrainingModeFlag = false;
    found = true;
  }

  if (!found) {
    GateError("Error in Gate_NN_ARF_Actor macro 'setMode', must be 'train' or test', while read " << m);
  }

  GateMessage("Actor", 1, "Gate_NN_ARF_Actor mode = " << m);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetNNModel(std::string& m)
{
  mNNModelPath = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetNNDict(std::string& m)
{
  mNNDictPath = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetMaxAngle(double a)
{
  mMaxAngle = a/deg;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetRRFactor(int f)
{
  mRRFactor = f;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetImage(std::string& m)
{
  mImagePath = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetSpacing(double m, int index)
{
  mSpacing[index] = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetSize(int m, int index)
{
  mSize[index] = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetCollimatorLength(double m)
{
  mCollimatorLength = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetBatchSize(double m)
{
  mBatchSize = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::Construct()
{
  GateDebugMessageInc("Actor", 4, "Gate_NN_ARF_Actor -- Construct - begin\n");
  GateVActor::Construct();

  /* Enable callbacks */
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

#ifdef GATE_USE_TORCH
  
  //Load the nn and the json dictionary
  
  if (mNNModelPath == "")
    GateError("Error: Neural Network model filename (.pt) is empty. Use setNNModel");
  
  if (mNNDictPath == "")
    GateError("Error: Neural Network dictionay filename (.json) is empty. Use setNNDict");

  // Load the neural network
  mNNModule = torch::jit::load(mNNModelPath);

  // No CUDA for the moment
  // mNNModule->to(torch::kCUDA);  //FIXME not cuda

  // Load the json file
  std::ifstream nnDictFile(mNNDictPath);
  using json = nlohmann::json;
  json nnDict;
  try {
    nnDictFile >> nnDict;
  } catch(std::exception & e) {
    GateError("Cannot open dict json file: " << mNNDictPath);
  }
  try {
    std::vector<double> tempXmean = nnDict["x_mean"];
    mXmean = tempXmean;
    std::vector<double> tempXstd = nnDict["x_std"];
    mXstd = tempXstd;
  } catch(std::exception & e) {
    GateError("Cannot find x_mean and x_std in the dict json file: " << mNNDictPath);
  }

  if (mRRFactor != 0) {
    GateError("setRussianRoulette option must NOT be used in test mode");
  }
  
  if (nnDict.find("rr") != nnDict.end())
    mRRFactor = nnDict["rr"];
  else
    mRRFactor = nnDict["RR"];

  if (mRRFactor == 0.0) {
    GateError("Cannot find RR value in the dict json file: " << mNNDictPath);
  }
  
  mNNOutput = at::empty({0,0});
  assert(mNNModule != nullptr);
#endif

  ResetData();
  GateMessageDec("Actor", 4, "Gate_NN_ARF_Actor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveData()
{
  GateVActor::SaveData(); // Need to change filename if ask by user (OoverwriteFileFlag)

  if (mTrainingModeFlag) {
    GateMessage("Actor", 1, "NN_ARF_Actor Number of Detected events: "
                << mNumberOfDetectedEvent
                << " / " << mTrainData.size()
                << " = " << (double)mNumberOfDetectedEvent/(double)mTrainData.size()*100.0
                << "%" << std::endl);
  }
  else {
    GateMessage("Actor", 1, "NN_ARF_Actor Number of Stored events: "
                << " " << mTestData.size() << std::endl);

  }
  GateMessage("Actor", 1, "NN_ARF_Actor Max angles: "
              << mThetaMax << " " << mPhiMax << std::endl);

  // Root Output
  mSaveFilename = mSaveFilename;

  auto pFile = new TFile(mSaveFilename, "RECREATE", "ROOT file Gate_NN_ARF_Actor", 9);
  if (mTrainingModeFlag) {
    auto pListeVar = new TTree("ARF (training)", "ARF Training Dataset");
    double t,p,e,w;//, weight;
    pListeVar->Branch("Theta", &t, "Theta/D");
    pListeVar->Branch("Phi", &p, "Phi/D");
    pListeVar->Branch("E", &e, "E/D");
    pListeVar->Branch("window", &w, "w/D");
    // We dont store the weight because it is easily retrieve: 1.0 everywhere
    // except if w == 0;
    //if (mRRFactor != 0)
    //  pListeVar->Branch("weight", &weight, "Weight/D");
    for(unsigned int i=0; i<mTrainData.size(); i++) {
      t = mTrainData[i].theta;
      p = mTrainData[i].phi;
      e = mTrainData[i].E;
      w = mTrainData[i].w;
      /*if (mRRFactor) {
        if (w ==0) weight = mRRFactor;
        else weight = 1.0;
        }
      */
      pListeVar->Fill();
    }
  }
  else {
#ifdef GATE_USE_TORCH
    // process remaining particules if the current batch is not complete
    if (mBatchInputs.size() > 0) {
      ProcessBatch();
      if (mNNOutput.sizes()[0] > 0) {
        for (unsigned int testIndex=0; testIndex < mNNOutput.sizes()[0]; ++testIndex) {
          mTestData[testIndex + mCurrentSaveNNOutput].nn = std::vector<double>(mNNOutput.sizes()[1]);
          for (unsigned int outputIndex=0; outputIndex < mNNOutput.sizes()[1]; ++outputIndex) {
            mTestData[testIndex + mCurrentSaveNNOutput].nn[outputIndex] = mNNOutput[testIndex][outputIndex].item<double>();
          }
        }
        mCurrentSaveNNOutput += mNNOutput.sizes()[0];
      }
    }
#endif

    if (mSaveFilename != "FilnameNotGivenForThisActor") {
      auto pListeVar = new TTree("ARF (testing)", "ARF Testing Dataset Tree");
      double x,y,t,p,e;
      pListeVar->Branch("X", &x, "X/D");
      pListeVar->Branch("Y", &y, "Y/D");
      pListeVar->Branch("Theta", &t, "Theta/D");
      pListeVar->Branch("Phi", &p, "Phi/D");
      pListeVar->Branch("E", &e, "E/D");

      for(unsigned int i=0; i<mTestData.size(); i++) {
        x = mTestData[i].x;
        y = mTestData[i].y;
        t = mTestData[i].theta;
        p = mTestData[i].phi;
        e = mTestData[i].E;
        pListeVar->Fill();
      }
    }

#ifdef GATE_USE_TORCH
    //Write the image thanks to the NN
    if (mTestData.size() != 0) {
      double nb_ene = mTestData[0].nn.size();
      G4ThreeVector resolution(mSize[0],
                               mSize[1],
                               nb_ene+1); // +1 because first empty slice
      G4ThreeVector imageSize(resolution[0]*mSpacing[0]/2.0,
                              resolution[1]*mSpacing[1]/2.0,
                              resolution[2]/2.0);
      mImage->SetResolutionAndHalfSize(resolution, imageSize);
      mImage->Allocate();
      mImage->Fill(0.0);
      for(unsigned int i=0; i<mTestData.size(); i++) {
        if (!mTestData[i].nn.empty()) {
          double tx = mCollimatorLength*cos(mTestData[i].theta * pi /180.0);
          double ty = mCollimatorLength*cos(mTestData[i].phi * pi /180.0);
          int u = round((mTestData[i].y + tx + mSize[0]*mSpacing[0]/2.0 - mSpacing[0]/2.0)/mSpacing[0]);
          int v = round((mTestData[i].x + ty + mSize[1]*mSpacing[1]/2.0 - mSpacing[1]/2.0)/mSpacing[1]);
          if (u < 0 || u > (mSize[0]-1))
            continue;
          if (v < 0 || v > (mSize[1]-1))
            continue;
          for (unsigned int energy=1; energy<nb_ene+1; ++energy) {
            mImage->SetValue(v, u, energy, mImage->GetValue(v, u, energy) + mTestData[i].nn[energy]/mNDataset);
          }
        }
      }
      mImage->Write(mImagePath);
      GateMessage("Actor", 1, "NN_ARF_Actor Projection written in " << mImagePath << G4endl);
      GateMessage("Actor", 1, "NN_ARF_Actor Number of energy windows " << nb_ene << G4endl);
      GateMessage("Actor", 1, "NN_ARF_Actor Number of events " << mNDataset << G4endl);
      GateMessage("Actor", 1, "NN_ARF_Actor Number of events reaching the detection plane " << mTestData.size() << G4endl);
      GateMessage("Actor", 1, "NN_ARF_Actor Number of batch " << mNumberOfBatch << G4endl);
    }
    else {
      GateMessage("Actor", 1, "NN_ARF_Actor No detected events, no image written." << std::endl << G4endl);
    }
#endif
  }
  pFile->Write();
  pFile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ResetData()
{
  mTrainData.clear();
  mTestData.clear();
  mNDataset = 0; // needed for normalization at the end
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::BeginOfRunAction(const G4Run * r)
{
  GateVActor::BeginOfRunAction(r);
  mNumberOfDetectedEvent = 0;
  if (mTrainingModeFlag) {
    G4DigiManager * fDM = G4DigiManager::GetDMpointer();
    for(auto name:mListOfWindowNames) {
      auto id = fDM->GetDigiCollectionID(name);
      GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> energy window " <<
                  name << " (id = " << id << ")" << std::endl);
      if (id == -1) {
        GateError("Cannot find the energy window named: " << name);
      }
      mListOfWindowIds.push_back(id);
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
  mNDataset++;
  mEventIsAlreadyStored = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::EndOfEventAction(const G4Event * e)
{
  static int russian_roulette_current = 0;
  GateVActor::EndOfEventAction(e);
  if (mTrainingModeFlag) {
    G4DigiManager * fDM = G4DigiManager::GetDMpointer();
    bool isIn = false;
    int i=0;
    for(auto id:mListOfWindowIds) {
      ++i;
      auto SDC = dynamic_cast<const GateSingleDigiCollection*>(fDM->GetDigiCollection(id));
      if (!SDC) continue;
      /*
      // ==> No need for u,v coordinates. Keep this comments for future ref.
      G4double xProj = (*SDC)[0]->GetLocalPos()[0];
      G4double yProj = (*SDC)[0]->GetLocalPos()[1];
      mCurrentOutData.u = xProj;
      mCurrentOutData.v = yProj;
      */
      if (mEnergyModeFlag) // Currently never true (experimental)
        mCurrentTrainData.w = (*SDC)[0]->GetEnergy();
      else
        mCurrentTrainData.w = i;

      isIn = true;
      ++mNumberOfDetectedEvent;
    }
    if (!isIn) {
      if (mRRFactor == 0 or russian_roulette_current == mRRFactor-1) {
        mCurrentTrainData.w = 0;                   // windows 0 is 'outside'
        russian_roulette_current = 0;              // reset current value
        mTrainData.push_back(mCurrentTrainData);   // store data
      }
      else { // ignore this event
        ++russian_roulette_current;
      }
    }
    else {
      mTrainData.push_back(mCurrentTrainData);
    }
  }// end mTrainingModeFlag
  else {
    // Do not count event that never go to UserSteppingAction
    if (mEventIsAlreadyStored and !mIgnoreCurrentData) {
      mTestData.push_back(mCurrentTestData);

#ifdef GATE_USE_TORCH
      if (mNNOutput.sizes()[0] > 0) {
        for (unsigned int testIndex=0; testIndex < mNNOutput.sizes()[0]; ++testIndex) {
          mTestData[testIndex + mCurrentSaveNNOutput].nn = std::vector<double>(mNNOutput.sizes()[1]);
          for (unsigned int outputIndex=0; outputIndex < mNNOutput.sizes()[1]; ++outputIndex) {
            mTestData[testIndex + mCurrentSaveNNOutput].nn[outputIndex] = mNNOutput[testIndex][outputIndex].item<double>();
          }
        }
        mCurrentSaveNNOutput += mNNOutput.sizes()[0];
      }
#endif
      
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::UserSteppingAction(const GateVVolume * /*v*/, const G4Step* step)
{
  if (mEventIsAlreadyStored) return;

  // Get coordinate in the current volume coordinate system
  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable());

  // Get information
  auto pre = step->GetPreStepPoint();
  //auto post = step->GetPostStepPoint();
  auto p = theTouchable->GetHistory()->GetTopTransform().TransformPoint(pre->GetPosition());
  auto E = pre->GetKineticEnergy();

  //The momentum direction is defined with respect to the world frame. Convert to detector frame.
  auto dir = pre->GetMomentumDirection();
  dir = theTouchable->GetHistory()->GetTopTransform().TransformAxis(dir);

  // Spherical coordinates
  // https://en.wikipedia.org/wiki/Spherical_coordinate_system
  // https://mathinsight.org/spherical_coordinates
  auto theta = acos(dir.y())/degree;
  auto phi = acos(dir.x())/degree;

  // Threshold on angles: do not store if larger (for debug)
  mIgnoreCurrentData = false;  
  if (mMaxAngle != 0.0 and
      (fabs(theta) > mMaxAngle or
       fabs(phi)   > mMaxAngle)) {
    mIgnoreCurrentData = true;
    mEventIsAlreadyStored = true;
    return;
  }
  
  if (mTrainingModeFlag) {
    mCurrentTrainData.E = E;
    mCurrentTrainData.theta = theta;
    mCurrentTrainData.phi = phi;
  }
  else {
    mCurrentTestData.x = p.x();
    mCurrentTestData.y = p.y();
    mCurrentTestData.E = E;
    mCurrentTestData.theta = theta;
    mCurrentTestData.phi = phi;

#ifdef GATE_USE_TORCH
    // Create a vector of input and push it in the bash inputs.
    // If batch inputs is full (size = mBatchSize) then pass it to the Neural Network
    // Else, get the next particle
    std::vector<double> tempVector {(theta - mXmean[0])/mXstd[0], (phi - mXmean[1])/mXstd[1], (E - mXmean[2])/mXstd[2]};
    mBatchInputs.push_back(tempVector);

    mNNOutput = at::empty({0,0});

    if (mBatchInputs.size() >= mBatchSize) ProcessBatch();
    
#endif
  }

  // Output will be set EndOfEventAction
  mEventIsAlreadyStored = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ProcessBatch()
{
#ifdef GATE_USE_TORCH
  GateMessage("Actor", 1, "NN_ARF_Actor process batch of "
              << mBatchInputs.size() << " particles" << G4endl);
  mNumberOfBatch++;
      
  //Convert NN inputs to Tensor
  std::vector<torch::jit::IValue> inputTensorContainer;
  torch::Tensor inputTensor = torch::zeros({(unsigned int)mBatchInputs.size(), 3});
  for (unsigned int inputIndex=0; inputIndex<mBatchInputs.size(); ++inputIndex) {
    inputTensor[inputIndex][0] = mBatchInputs[inputIndex][0];
    inputTensor[inputIndex][1] = mBatchInputs[inputIndex][1];
    inputTensor[inputIndex][2] = mBatchInputs[inputIndex][2];
  }
  //inputTensorContainer.push_back(inputTensor.cuda());
  inputTensorContainer.push_back(inputTensor); // NOT CUDA

  // Execute the model and turn its output into a tensor.
  mNNOutput = mNNModule->forward(inputTensorContainer).toTensor();

  // Normalize output
  mNNOutput = exp(mNNOutput);
  for (unsigned int tensorIndex=0; tensorIndex < mNNOutput.sizes()[0]; ++tensorIndex) {
    mNNOutput[tensorIndex] = mNNOutput[tensorIndex]/sum(mNNOutput[tensorIndex]);
  }

  // Normalize with russian roulette
  for (unsigned int outputIndex=0; outputIndex < mNNOutput.sizes()[0]; ++outputIndex) {
    mNNOutput[outputIndex][0] *= mRRFactor;
  }
  for (unsigned int tensorIndex=0; tensorIndex < mNNOutput.sizes()[0]; ++tensorIndex) {
    mNNOutput[tensorIndex] = mNNOutput[tensorIndex]/sum(mNNOutput[tensorIndex]);
  }

  // Clean the inputs
  mBatchInputs.clear();
#endif
}
//-----------------------------------------------------------------------------
