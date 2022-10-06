/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPHASESPACESOURCE_HH
#define GATEPHASESPACESOURCE_HH

#include "GateConfiguration.h"

#ifdef GATE_USE_TORCH
// Need to be *before* include GateIAEAHeader because it define macros
// that mess with torch
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"

#include <torch/script.h>

#pragma GCC diagnostic pop
#endif

#include "G4Event.hh"
#include "globals.hh"
#include "G4VPrimaryGenerator.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleMomentum.hh"
#include <iomanip>
#include <vector>

#include "GateVSource.hh"
#include "GateSourcePhaseSpaceMessenger.hh"
#include "GateUserActions.hh"
#include "GateTreeFileManager.hh"
#include <typeindex>
#include "json.hpp"

struct iaea_record_type;
struct iaea_header_type;

class GateSourcePhaseSpace : public GateVSource
{
public:
    GateSourcePhaseSpace(G4String name);

    ~GateSourcePhaseSpace();

    void Initialize();

    void GenerateROOTVertex(G4Event *);

    void GenerateROOTVertexSingle();

    void GenerateROOTVertexPairs();

    void GenerateIAEAVertex(G4Event *);

    void GeneratePyTorchVertex(G4Event *);

    void GeneratePyTorchVertexSingle(G4Event *);

    void GeneratePyTorchVertexPairs(G4Event *);

    void GenerateBatchSamplesFromPyTorchSingle();

    void GenerateBatchSamplesFromPyTorchPairs();

    G4int OpenIAEAFile(G4String file);

    G4int GeneratePrimaries(G4Event *event);

    void GeneratePrimariesSingle(G4Event *event);

    void GeneratePrimariesPairs(G4Event *event);

    void GenerateVertex(G4Event *event, G4ThreeVector &position,
                        G4ThreeVector &momentum, double time, double w);

    void UpdatePositionAndMomentum(G4ThreeVector &position, G4ThreeVector &momentum);

    void AddFile(G4String file);

    void InitializeTransformation();

    G4ThreeVector SetReferencePosition(G4ThreeVector coordLocal);

    G4ThreeVector SetReferenceMomentum(G4ThreeVector coordLocal);

    void SetRelativeTimeFlag(bool b);

    void SetIgnoreTimeFlag(bool b);

    bool GetPositionInWorldFrame() const { return mPositionInWorldFrame; }

    void SetPositionInWorldFrame(bool t) { mPositionInWorldFrame = t; }

    void SetUseRegularSymmetry()
    {
        if (mUseRandomSymmetry)
            GateError("You cannot use random and regular symmetry for phase space source");
        mUseRegularSymmetry = true;
    }

    bool GetUseRegularSymmetry() const { return mUseRegularSymmetry; }

    void SetUseRandomSymmetry()
    {
        if (mUseRegularSymmetry)
            GateError("You cannot use random and regular symmetry for phase space source");
        mUseRandomSymmetry = true;
    }

    bool GetUseRandomSymmetry() const { return mUseRandomSymmetry; }

    void SetParticleType(G4String &name) { mParticleTypeNameGivenByUser = name; }

    void SetParticlePDGCode(G4int code) { mPDGCodeGivenByUser = code; }

    void SetUseNbOfParticleAsIntensity(bool b) { mUseNbOfParticleAsIntensity = b; }

    void SetStartingParticleId(long id) { mStartingParticleId = id; }

    void SetIgnoreWeight(bool b) { mIgnoreWeight = b; }

    void SetPytorchBatchSize(int b) { mPTBatchSize = b; }

    void InitializeIAEA();

    void InitializeROOT();

    void InitializeROOTSingle();

    void InitializeROOTPairs();

    void InitializePyTorch();

    void InitializePyTorchPairs();

    void InitializePyTorchSingle();

    void SetPytorchParams(G4String &name) { mPTJsonFilename = name; }

protected:
    std::vector<unsigned int> pListOfSelectedEvents;

    G4long mCurrentParticleNumber;
    G4long mCurrentParticleNumberInFile;
    G4long mStartingParticleId;
    G4long mNumberOfParticlesInFile;
    G4long mTotalNumberOfParticles;
    G4int mCurrentRunNumber;
    double mTotalSimuTime;
    double mRequestedNumberOfParticlesPerRun;
    G4long mLoop;
    G4long mLoopFile;
    G4long mCurrentUse;
    G4long mResidu;
    double mResiduRun;
    G4long mLastPartIndex;
    unsigned int mCurrentParticleInIAEAFiles;
    G4long mCurrentUsedParticleInIAEAFiles;
    bool mInitialized;
    G4String mFileType;
    bool mRelativeTimeFlag;
    bool mTimeIsUsed;

    float energy;
    float x, y, z;
    float dx, dy, dz;
    float ftime;
    double dtime;
    std::type_index time_type = typeid(nullptr);
    float weight;

    // below variables for pairs of events
    // FIXME float or double ?
    bool mIsPair;
    float E1;
    float E2;
    float X1;
    float Y1;
    float Z1;
    float X2;
    float Y2;
    float Z2;
    float dX1;
    float dY1;
    float dZ1;
    float dX2;
    float dY2;
    float dZ2;
    float t1;
    float t2;
    float w1;
    float w2;

    G4int mPDGCode;
    G4int mPDGCodeGivenByUser;
    char particleName[64];
    G4String mParticleTypeNameGivenByUser;
    double mParticleTime;
    G4double mMomentum;

    bool mAlreadyLoad;

    double px;
    double py;
    double pz;

    std::vector<G4String> listOfPhaseSpaceFile;

    bool mPositionInWorldFrame;

    FILE *pIAEAFile;
    iaea_record_type *pIAEARecordType;
    iaea_header_type *pIAEAheader;

    G4ParticleDefinition *pParticleDefinition;
    G4PrimaryParticle *pParticle;
    G4PrimaryVertex *pVertex;
    G4ThreeVector mParticlePosition;
    G4ThreeVector mParticleMomentum;

    G4ThreeVector mParticlePositionPair1;
    G4ThreeVector mParticlePositionPair2;
    G4ThreeVector mParticleMomentumPair1;
    G4ThreeVector mParticleMomentumPair2;

    std::vector<const G4RotationMatrix *> mListOfRotation;
    std::vector<G4ThreeVector> mListOfTranslation;

    bool mUseRegularSymmetry;
    bool mUseRandomSymmetry;
    double mAngle;

    bool mUseNbOfParticleAsIntensity;
    GateInputTreeFileChain mChain;

    bool mIgnoreWeight;

    int mPTCurrentIndex;
    // requested batch size
    int mPTBatchSize;
    // real umber of generated particle in the batch (can be inferior to mPTBatchSize)
    int mPTCurrentBatchSize;
    double mPTmass;
    std::vector<G4ThreeVector> mPTPosition;
    std::vector<double> mPTDX;
    std::vector<double> mPTDY;
    std::vector<double> mPTDZ;
    std::vector<double> mPTEnergy;

    nlohmann::json mPTParam;
    std::string mPTJsonFilename;
    bool mPTnormalize;
    std::vector<double> mPT_x_mean;
    std::vector<double> mPT_x_std;
    std::vector<std::string> mPT_keys;
    int mPTz_dim;
    std::map<std::string, double> mDefaultKeyValues;

    int get_key_index(std::string key);

    double get_key_value(const float *v, int i, double def);

    int mPTEnergyIndex;
    int mPTPositionXIndex;
    int mPTPositionYIndex;
    int mPTPositionZIndex;
    int mPTDirectionXIndex;
    int mPTDirectionYIndex;
    int mPTDirectionZIndex;

    int mPTPositionIndex[2][3];
    int mPTDirectionIndex[2][3];
    int mPTEnergiesIndex[2];
    int mPTTimeIndex[2];

    std::vector<G4ThreeVector> mPTPositions[2];
    std::vector<G4ThreeVector> mPTDirections[2];
    std::vector<double> mPTEnergies[2];
    std::vector<double> mPTTimes[2];

#ifdef GATE_USE_TORCH
    torch::jit::script::Module mPTmodule;
    torch::Tensor mPTzer;
#endif
};

#endif
