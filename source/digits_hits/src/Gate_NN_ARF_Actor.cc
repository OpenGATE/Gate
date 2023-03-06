/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "json.hpp"
#include "Gate_NN_ARF_Actor.hh"
#include "GateDigi.hh"
#include "GateTreeFileManager.hh"
#include "G4DigiManager.hh"
#include "TTree.h"
#include <iostream>

#ifdef GATE_USE_TORCH
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"

#include <torch/torch.h>

#pragma GCC diagnostic pop
#endif

//-----------------------------------------------------------------------------
void Gate_NN_ARF_Train_Data::Print(std::ostream &os) const {
    os << " train = "
       << theta << " "
       << phi << " "
       << E << " : "
       << w << " "
       << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void Gate_NN_ARF_Predict_Data::Print(std::ostream &os) const {
    os << " test = "
       << x << " "
       << y << " "
       << theta << " "
       << phi << " "
       << E << " "
       << copy_id << " "
       << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::Gate_NN_ARF_Actor(G4String name, G4int depth) :
        GateVActor(name, depth) {
    GateDebugMessageInc("Actor", 4, "Gate_NN_ARF_Actor() -- begin\n");
    pMessenger = new Gate_NN_ARF_ActorMessenger(this);
    mARFMode = "predict";
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
    GateDebugMessageDec("Actor", 4, "Gate_NN_ARF_Actor() -- end\n");
    mNNModelPath = "";
    mNNDictPath = "";
    mNumberOfBatch = 0;
    mListModeOutputFilename = "";
    mARFOutputFilename = "";
    mSquaredOutputFlag = false;
    mNumberOfCopies = 0;
    mImage = new GateImageDouble();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::~Gate_NN_ARF_Actor() {
    delete pMessenger;
    delete mImage;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetEnergyWindowNames(std::string &names) {
    std::vector<std::string> words;
    GetWords(words, names);
    for (const auto &w:words) mListOfWindowNames.push_back(w);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetMode(std::string m) {
    bool found = false;
    if (m == "train") {
        mARFMode = "train";
        found = true;
    }
    if (m == "predict" or m == "test") {
        // we keep 'test' for backward compatibility
        mARFMode = "predict";
        found = true;
#ifndef GATE_USE_TORCH
        GateError("Error: cannot use 'predict' mode with NN_ARF_Actor, GATE must be compiled with USE_TORCH.");
#endif
    }
    if (!found) {
        GateError("Error in Gate_NN_ARF_Actor macro 'setMode', "
                  "must be 'train' or 'predict', while I read " << m);
    }
    GateMessage("Actor", 1, "Gate_NN_ARF_Actor mode = " << m);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetNNModel(std::string &m) {
    mNNModelPath = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetNNDict(std::string &m) {
    mNNDictPath = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetMaxAngle(double a) {
    mMaxAngle = a / deg;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetRRFactor(int f) {
    mRRFactor = f;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::EnableSquaredOutput(bool b) {
    mSquaredOutputFlag = b;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetListModeOutputFilename(std::string &m) {
    mListModeOutputFilename = m;
    GateError("List Mode output not yet implemented, sorry.");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetARFOutputFilename(std::string &m) {
    mARFOutputFilename = m;
    GateError("ARF output not yet implemented, sorry.");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetSpacing(double m, int index) {
    mSpacing[index] = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetSize(int m, int index) {
    mSize[index] = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetCollimatorLength(double m) {
    mCollimatorLength = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetBatchSize(double m) {
    mBatchSize = m;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::Construct() {
    GateDebugMessageInc("Actor", 4, "Gate_NN_ARF_Actor -- Construct - begin\n");
    GateVActor::Construct();

    // Enable callbacks
    EnableBeginOfRunAction(true);
    EnableBeginOfEventAction(true);
    EnableEndOfEventAction(true);
    EnablePreUserTrackingAction(false);
    EnableUserSteppingAction(true);

    // Check
    G4String extension = getExtension(mSaveFilename);
    if (mARFMode == "train") {
        if (extension != "root")
            GateError("Only use root output for 'train' mode");
    }
    if (mARFMode == "predict") {
        if (extension != "mhd")
            GateError("Only use mhd output for 'predict' mode");
    }

#ifdef GATE_USE_TORCH
    if (mARFMode == "predict") {
        // Load the nn and the json dictionary
        if (mNNModelPath == "")
            GateError("Error: Neural Network model filename (.pt) is empty. Use setNNModel");
        if (mNNDictPath == "")
            GateError("Error: Neural Network dictionary filename (.json) is empty. Use setNNDict");
        mNNModule = torch::jit::load(mNNModelPath);

        // No CUDA for the moment
        // mNNModule.to(torch::kCUDA);  //FIXME no cuda yet

        // Load the json file
        std::ifstream nnDictFile(mNNDictPath);
        using json = nlohmann::json;
        json nnDict;
        try {
            nnDictFile >> nnDict;
        } catch (std::exception &e) {
            GateError("Cannot open dict json file: " << mNNDictPath);
        }
        try {
            std::vector<double> tempXmean = nnDict["x_mean"];
            mXmean = tempXmean;
            std::vector<double> tempXstd = nnDict["x_std"];
            mXstd = tempXstd;
        } catch (std::exception &e) {
            GateError("Cannot find x_mean and x_std in the dict json file: " << mNNDictPath);
        }

        if (mRRFactor != 0) {
            GateError("setRussianRoulette option must NOT be used in 'predict' mode");
        }

        if (nnDict.find("rr") != nnDict.end())
            mRRFactor = nnDict["rr"];
        else
            mRRFactor = nnDict["RR"];

        if (mRRFactor == 0.0) {
            GateError("Cannot find RR value in the dict json file: " << mNNDictPath);
        }

        mNNOutput = at::empty({0, 0});
        //assert(mNNModule != nullptr);
    }
#endif

    // Repeated volume ?
    auto stop = false;
    auto vol = mVolume;
    mVolumeDepth = 0;
    auto depth = 0;
    mNumberOfCopies = 1;
    while (not stop) {
        depth += 1;
        vol = vol->GetMotherCreator();
        if (vol->GetVolumeNumber() > 1) {
            if (mNumberOfCopies > 1)
                GateError("Several ancestors of "
                                  << mVolume->GetObjectName()
                                  << " are repeated, only one is allowed.");
            mNumberOfCopies = vol->GetVolumeNumber();
            mVolumeDepth = depth;
        }
        if (vol->GetObjectName() == "world") stop = true;
    }

    ResetData();
    GateMessageDec("Actor", 4, "Gate_NN_ARF_Actor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveData() {
    // Needed to change filename if ask by user (OverwriteFileFlag)
    GateVActor::SaveData();
    if (mARFMode == "train") SaveDataTrainMode();
    else SaveDataPredictMode();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveDataTrainMode() {
    GateMessage("Actor", 1, "NN_ARF_Actor Number of Detected events: "
            << mNumberOfDetectedEvent
            << " / " << mTrainData.size()
            << " = " << (double) mNumberOfDetectedEvent / (double) mTrainData.size() * 100.0
            << "%" << std::endl);

    GateMessage("Actor", 1, "NN_ARF_Actor Max angles: "
            << mThetaMax << " " << mPhiMax << std::endl);

    // Write the tree (root or npy)
    G4String extension = getExtension(mSaveFilename);
    GateOutputTreeFileManager mFile;
    if (extension == "root") mFile.add_file(mSaveFilename, "root");

    mFile.add_file(mSaveFilename, extension);
    mFile.set_tree_name("ARF (training)");
    double t, p, e, w;
    mFile.write_variable("Theta", &t);
    mFile.write_variable("Phi", &p);
    mFile.write_variable("E", &e);
    mFile.write_variable("window", &w);
    mFile.write_header();
    // Later: may be interesting to store the weight

    for (unsigned int i = 0; i < mTrainData.size(); i++) {
        t = mTrainData[i].theta;
        p = mTrainData[i].phi;
        e = mTrainData[i].E;
        w = mTrainData[i].w;
        mFile.fill();
    }
    mFile.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveDataPredictMode() {

#ifdef GATE_USE_TORCH
    ProcessBatch();
    ProcessBatchEnd();

    // Check that some data have been predicted
    if (mPredictData.size() == 0 or mPredictData[0].nn.size() == 0) {
        GateWarning("NN_ARF_Actor has no detected event, write nothing.");
        return;
    }

    // Split data according to copy_id
    if (mNumberOfCopies == 1) SaveDataProjection(-1);
    else {
        for (int cp = 0; cp < mNumberOfCopies; cp++) {
            SaveDataProjection(cp);
        }
    }

    double nb_ene = mPredictData[0].nn.size();
    GateMessage("Actor", 1, "NN_ARF_Actor Projection written in " << mSaveFilename << G4endl);
    GateMessage("Actor", 1, "NN_ARF_Actor Number of energy windows " << nb_ene << G4endl);
    GateMessage("Actor", 1, "NN_ARF_Actor Number of events " << mNDataset << G4endl);
    GateMessage("Actor", 1,
                "NN_ARF_Actor Number of events reaching the detection plane " << mPredictData.size() << G4endl);
    GateMessage("Actor", 1, "NN_ARF_Actor Number of batch " << mNumberOfBatch << G4endl);

#endif

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveDataProjection(int cp) {

    // Define output filename (add run_id and copy_id if needed)
    auto filename = G4String(mSaveFilename);
    if (cp != -1) {
        // if the copy_nb is -1, it means one single copy_nb,
        // no need to append it to the filename
        auto extension = getExtension(filename);
        filename = removeExtension(filename);
        filename = filename + "_head_" + std::to_string(cp);
        filename = filename + "." + extension;
    }

    // Write the image thanks to the NN
    double nb_ene = mPredictData[0].nn.size();
    G4ThreeVector resolution(mSize[0],
                             mSize[1],
                             nb_ene); // +1 because first channel is an empty slice ?
    G4ThreeVector imageSize(resolution[0] * mSpacing[0] / 2.0,
                            resolution[1] * mSpacing[1] / 2.0,
                            resolution[2] / 2.0);
    mImage->SetResolutionAndHalfSize(resolution, imageSize);
    mImage->Allocate();
    mImage->Fill(0.0);

    // Squared image ?
    auto mImageSquared = new GateImageDouble();
    if (mSquaredOutputFlag) {
        mImageSquared->SetResolutionAndHalfSize(resolution, imageSize);
        mImageSquared->Allocate();
        mImageSquared->Fill(0.0);
    }

    // Loop on event
    int id = cp;
    if (cp == -1) id = 0;
    for (unsigned int i = 0; i < mPredictData.size(); i++) {
        if (mPredictData[i].copy_id == id and !mPredictData[i].nn.empty()) {
            double tx = mCollimatorLength * cos(mPredictData[i].theta * pi / 180.0);
            double ty = mCollimatorLength * cos(mPredictData[i].phi * pi / 180.0);
            int u = round((mPredictData[i].y + tx + mSize[0] * mSpacing[0] / 2.0 - mSpacing[0] / 2.0) / mSpacing[0]);
            int v = round((mPredictData[i].x + ty + mSize[1] * mSpacing[1] / 2.0 - mSpacing[1] / 2.0) / mSpacing[1]);
            if (u < 0 || u > (mSize[0] - 1))
                continue;
            if (v < 0 || v > (mSize[1] - 1))
                continue;
            for (unsigned int energy = 1; energy < nb_ene; ++energy) {
                auto val = mPredictData[i].nn[energy];
                auto value = mImage->GetValue(v, u, energy) + val;
                mImage->SetValue(v, u, energy, value);
                if (mSquaredOutputFlag) {
                    auto valuesq = mImageSquared->GetValue(v, u, energy) + val * val;
                    mImageSquared->SetValue(v, u, energy, valuesq);
                }
            }
        }
    }

    // scale per events
    for (auto p = mImage->begin(); p < mImage->end(); p++) *p /= mNDataset;
    if (mSquaredOutputFlag)
        for (auto p = mImageSquared->begin(); p < mImageSquared->end(); p++) *p /= mNDataset;

    // write
    mImage->Write(filename);
    if (mSquaredOutputFlag) {
        auto mImagePathSquared = removeExtension(filename) + "-Squared.mhd";
        mImageSquared->Write(mImagePathSquared);
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveDataListmode() {
    GateError("List Mode output not yet implemented, sorry.");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveDataARF() {
    GateError("ARF output not yet implemented, sorry.");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ResetData() {
    mTrainData.clear();
    mPredictData.clear();
    mBatchInputs.clear();
    mListOfWindowIds.clear();

#ifdef GATE_USE_TORCH
    mNNOutput = at::empty({0, 0});
#endif

    mNumberOfBatch = 0;
    mCurrentSaveNNOutput = 0;
    mNDataset = 0; // needed for normalization at the end
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::BeginOfRunAction(const G4Run *r) {
    GateVActor::BeginOfRunAction(r);
    mNumberOfDetectedEvent = 0;
    if (mARFMode == "train") {
        G4DigiManager *fDM = G4DigiManager::GetDMpointer();
        for (auto name:mListOfWindowNames) {
            auto id = fDM->GetDigiCollectionID(name);
            GateMessage("Actor", 1,
                        "Gate_NN_ARF_Actor -> energy window "
                                << name << " (id = " << id << ")" << std::endl);
            if (id == -1) {
                GateError("Cannot find the energy window named: " << name);
            }
            mListOfWindowIds.push_back(id);
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::BeginOfEventAction(const G4Event *e) {
    GateVActor::BeginOfEventAction(e);
    mNDataset++;
    mEventIsAlreadyStored = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::EndOfEventAction(const G4Event *e) {
    static int russian_roulette_current = 0;
    GateVActor::EndOfEventAction(e);
    if (mARFMode == "train") {
        G4DigiManager *fDM = G4DigiManager::GetDMpointer();
        bool isIn = false;
        int i = 0;
        for (auto id:mListOfWindowIds) {
            ++i;
            // OK GND 2022 TODO : check and adapt correct DigiCollection ID
            auto SDC = dynamic_cast<const GateDigiCollection *>(fDM->GetDigiCollection(id));
            if (!SDC) continue;
            /*
            // ==> No need for u,v coordinates. Keep this comments for future ref.
            G4double xProj = (*SDC)[0]->GetLocalPos()[0];
            G4double yProj = (*SDC)[0]->GetLocalPos()[1];
            mCurrentOutData.u = xProj;
            mCurrentOutData.v = yProj;
            */
            // Keep this for later: it may be interesting to use the energy, not only the channel
            // mCurrentTrainData.w = (*SDC)[0]->GetEnergy();
            mCurrentTrainData.w = i;

            isIn = true;
            ++mNumberOfDetectedEvent;
        }
        if (!isIn) {
            if (mRRFactor == 0 or russian_roulette_current == mRRFactor - 1) {
                mCurrentTrainData.w = 0;                   // windows 0 is 'outside'
                russian_roulette_current = 0;              // reset current value
                mTrainData.push_back(mCurrentTrainData);   // store data
            } else { // ignore this event
                ++russian_roulette_current;
            }
        } else {
            mTrainData.push_back(mCurrentTrainData);
        }
    }// end mARFMode train
    else {
        // Do not count event that never go to UserSteppingAction
        if (mEventIsAlreadyStored and !mIgnoreCurrentData) {
            mPredictData.push_back(mCurrentPredictData);
            ProcessBatchEnd();
        }
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::UserSteppingAction(const GateVVolume * /*v*/, const G4Step *step) {
    if (mEventIsAlreadyStored) return;

    // Get coordinate in the current volume coordinate system
    G4TouchableHistory *theTouchable = (G4TouchableHistory *) (step->GetPreStepPoint()->GetTouchable());

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
    auto theta = acos(dir.y()) / degree;
    auto phi = acos(dir.x()) / degree;

    // Threshold on angles: do not store if larger (for debug)
    mIgnoreCurrentData = false;
    if (mMaxAngle != 0.0 and
        (fabs(theta) > mMaxAngle or
         fabs(phi) > mMaxAngle)) {
        mIgnoreCurrentData = true;
        mEventIsAlreadyStored = true;
        return;
    }

    if (mARFMode == "train") {
        mCurrentTrainData.E = E;
        mCurrentTrainData.theta = theta;
        mCurrentTrainData.phi = phi;
    } else {
        mCurrentPredictData.x = p.x();
        mCurrentPredictData.y = p.y();
        mCurrentPredictData.E = E;
        mCurrentPredictData.theta = theta;
        mCurrentPredictData.phi = phi;
        if (mNumberOfCopies > 1)
            mCurrentPredictData.copy_id = pre->GetTouchableHandle()->GetCopyNumber(mVolumeDepth);
        else
            mCurrentPredictData.copy_id = 0;

#ifdef GATE_USE_TORCH
        // Create a vector of input and push it in the bash inputs.
        // If batch inputs is full (size = mBatchSize) then pass it to the Neural Network
        // Else, get the next particle
        std::vector<double> tempVector{(theta - mXmean[0]) / mXstd[0],
                                       (phi - mXmean[1]) / mXstd[1],
                                       (E - mXmean[2]) / mXstd[2]};
        mBatchInputs.push_back(tempVector);
        mNNOutput = at::empty({0, 0});
        if (mBatchInputs.size() >= mBatchSize) ProcessBatch();
#endif
    }

    // Output will be set EndOfEventAction
    mEventIsAlreadyStored = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ProcessBatch() {
#ifdef GATE_USE_TORCH
    GateMessage("Actor", 1, "NN_ARF_Actor process batch of "
            << mBatchInputs.size() << " particles" << G4endl);
    mNumberOfBatch++;

    //Convert NN inputs to Tensor
    std::vector<torch::jit::IValue> inputTensorContainer;
    torch::Tensor inputTensor = torch::zeros({(unsigned int) mBatchInputs.size(), 3});
    for (unsigned int inputIndex = 0; inputIndex < mBatchInputs.size(); ++inputIndex) {
        inputTensor[inputIndex][0] = mBatchInputs[inputIndex][0];
        inputTensor[inputIndex][1] = mBatchInputs[inputIndex][1];
        inputTensor[inputIndex][2] = mBatchInputs[inputIndex][2];
    }
    //inputTensorContainer.push_back(inputTensor.cuda());
    inputTensorContainer.push_back(inputTensor); // NOT CUDA

    // Execute the model and turn its output into a tensor.
    torch::NoGradGuard no_grad_guard;
    mNNOutput = mNNModule.forward(inputTensorContainer).toTensor();

    // Normalize output
    mNNOutput = exp(mNNOutput);
    for (unsigned int tensorIndex = 0; tensorIndex < mNNOutput.sizes()[0]; ++tensorIndex) {
        mNNOutput[tensorIndex] = mNNOutput[tensorIndex] / sum(mNNOutput[tensorIndex]);
    }

    // Normalize with russian roulette
    for (unsigned int outputIndex = 0; outputIndex < mNNOutput.sizes()[0]; ++outputIndex) {
        mNNOutput[outputIndex][0] *= mRRFactor;
    }
    for (unsigned int tensorIndex = 0; tensorIndex < mNNOutput.sizes()[0]; ++tensorIndex) {
        mNNOutput[tensorIndex] = mNNOutput[tensorIndex] / sum(mNNOutput[tensorIndex]);
    }

    // Clean the inputs
    mBatchInputs.clear();
#endif
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ProcessBatchEnd() {
#ifdef GATE_USE_TORCH
    if (mNNOutput.sizes()[0] > 0) {
        for (unsigned int testIndex = 0; testIndex < mNNOutput.sizes()[0]; ++testIndex) {
            mPredictData[testIndex + mCurrentSaveNNOutput].nn = std::vector<double>(mNNOutput.sizes()[1]);
            for (unsigned int outputIndex = 0; outputIndex < mNNOutput.sizes()[1]; ++outputIndex) {
                mPredictData[testIndex +
                             mCurrentSaveNNOutput].nn[outputIndex] = mNNOutput[testIndex][outputIndex].item<double>();
            }
        }
        mCurrentSaveNNOutput += mNNOutput.sizes()[0];
    }
#endif
}
//-----------------------------------------------------------------------------
