/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <torch/script.h>
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
  GateDebugMessageDec("Actor",4,"Gate_NN_ARF_Actor() -- end\n");
  mNNModelPath = "";
  mNNDictPath = "";
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::~Gate_NN_ARF_Actor()
{
  delete pMessenger;
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

  ResetData();
  GateMessageDec("Actor", 4, "Gate_NN_ARF_Actor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveData()
{
  GateVActor::SaveData(); // Need to change filename if ask by user (OoverwriteFileFlag)

  if (mTrainingModeFlag) {
    GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> Detected "
                << mNumberOfDetectedEvent
                << " / " << mTrainData.size()
                << " = " << (double)mNumberOfDetectedEvent/(double)mTrainData.size()*100.0
                << "%" << std::endl);
  }
  else {
    GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> Stored "
                << " " << mTestData.size() << std::endl);

  }
  GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> max angles "
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
  pFile->Write();
  pFile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ResetData()
{
  mTrainData.clear();
  mTestData.clear();
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

    //Load the nn and the json dictionary
    if (mNNModelPath == "")
      GateError("Error: Neural Network model path (.pt) is empty");
    if (mNNDictPath == "")
      GateError("Error: Neural Network dictionay path (.json) is empty");
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(mNNModelPath);
    module->to(torch::kCUDA);
    std::ifstream nnDictFile(mNNDictPath);
    using json = nlohmann::json;
    json nnDict;
    nnDictFile >> nnDict;
    std::vector<double> x_mean = nnDict["x_mean"];
    std::vector<double> x_std = nnDict["x_std"];

    assert(module != nullptr);

    //pass these data to nn
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input = torch::zeros({1, 3});
    input[0][0] = (theta - x_mean[0])/x_std[0];
    input[0][1] = (phi - x_mean[1])/x_std[1];
    input[0][2] = (E - x_mean[2])/x_std[2];
    inputs.push_back(input.cuda());

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module->forward(inputs).toTensor();

    //Display elements
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    //for (unsigned int outputIndex=0; outputIndex < output.sizes()[1]; ++outputIndex) {
    //  std::cout << output[0][outputIndex] << " ";
    //}
    //std::cout << std::endl << output.sizes()[0] << " " << output.sizes()[1] << std::endl;


  }

  // Output will be set EndOfEventAction
  mEventIsAlreadyStored = true;
}
//-----------------------------------------------------------------------------
