/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_TORCH
// Need to be *before* include GateIAEAHeader because it define macros
// that mess with torch
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"

#include <torch/script.h>
#include "json.hpp"

#pragma GCC diagnostic pop
#endif

#include "GateSourcePhaseSpace.hh"
#include "GateIAEAHeader.h"
#include "GateIAEARecord.h"
#include "GateIAEAUtilities.h"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Neutron.hh"
#include "G4Proton.hh"
#include "GateVVolume.hh"
#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"
#include "GateFileExceptions.hh"
#include <chrono>
#include <algorithm>
#include <iterator>
#include <sstream>

typedef unsigned int uint;

// ----------------------------------------------------------------------------------
GateSourcePhaseSpace::GateSourcePhaseSpace(G4String name) : GateVSource(name)
{
    mCurrentParticleNumber = 0;
    mNumberOfParticlesInFile = 0;
    mTotalNumberOfParticles = 0;
    mCurrentParticleNumberInFile = 0;
    mResidu = 0;
    mResiduRun = 0.;
    mLastPartIndex = 0;
    mParticleTypeNameGivenByUser = "none";
    mUseNbOfParticleAsIntensity = false;
    mStartingParticleId = 0;
    mCurrentRunNumber = -1;
    mLoop = 0;
    mLoopFile = 0;
    mCurrentUse = 0;
    mIgnoreWeight = false;
    mUseRegularSymmetry = false;
    mUseRandomSymmetry = false;
    mAngle = 0.;
    mPositionInWorldFrame = false;
    mRequestedNumberOfParticlesPerRun = 0;
    mInitialized = false;
    m_sourceMessenger = new GateSourcePhaseSpaceMessenger(this);
    mFileType = "";
    mParticleTime = 0.;
    pIAEAFile = nullptr;
    pIAEARecordType = nullptr;
    pIAEAheader = nullptr;
    pParticleDefinition = nullptr;
    pParticle = nullptr;
    pVertex = nullptr;
    mMomentum = 0.;
    mParticlePosition = G4ThreeVector();
    mParticleMomentum = G4ThreeVector();
    x = y = z = dx = dy = dz = px = py = pz = energy = 0.;
    dtime = -1.;
    ftime = -1.;
    weight = 1.;
    strcpy(particleName, "");
    mPTBatchSize = 1e5;
    mPTCurrentIndex = mPTBatchSize;
    mTotalSimuTime = 0.;
    mAlreadyLoad = false;
    mCurrentParticleInIAEAFiles = 0;
    mCurrentUsedParticleInIAEAFiles = 0;
    mIsPair = false;
    mRelativeTimeFlag = false;
    mTimeIsUsed = true;
    mPDGCode = 0; // 0 is generic ion
    mPDGCodeGivenByUser = 0;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
GateSourcePhaseSpace::~GateSourcePhaseSpace()
{
    listOfPhaseSpaceFile.clear();
    if (pIAEAFile)
        fclose(pIAEAFile);
    pIAEAFile = nullptr;
    free(pIAEAheader);
    free(pIAEARecordType);
    pIAEAheader = nullptr;
    pIAEARecordType = nullptr;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::Initialize()
{
    InitializeTransformation();
    mTotalSimuTime =
        GateApplicationMgr::GetInstance()->GetTimeStop() - GateApplicationMgr::GetInstance()->GetTimeStart();

    mInitialized = false;
    if (mFileType == "IAEAFile")
        InitializeIAEA();
    if (mFileType == "pytorch")
        InitializePyTorch();
    if (mFileType == "root")
        InitializeROOT();

    if (!mInitialized)
    {
        DD(mFileType);
        GateError("Cannot initialize source-phsp. Wrong mFileType?");
    }

    if (mUseNbOfParticleAsIntensity)
        SetIntensity(mNumberOfParticlesInFile);

    GateMessage("Beam", 1, "Phase Space Source. Total nb of particles in PhS " << mNumberOfParticlesInFile << Gateendl);
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializeIAEA()
{
    int totalEventInFile = 0;
    mCurrentParticleNumberInFile = -1;
    G4String IAEAFileName = " ";
    for (const auto &j : listOfPhaseSpaceFile)
    {
        IAEAFileName = G4String(removeExtension(j));
        totalEventInFile = OpenIAEAFile(IAEAFileName);
        mTotalNumberOfParticles += totalEventInFile;
    }
    mInitialized = true;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializeROOT()
{
    for (const auto &file : listOfPhaseSpaceFile)
    {
        GateMessage("Beam", 1, "Phase Space Source. Read file " << file << Gateendl);
        auto extension = getExtension(file);
        mChain.add_file(file, extension);
    }
    mChain.set_tree_name("PhaseSpace");
    mChain.read_header();

    mTotalNumberOfParticles = mChain.nb_elements();
    mNumberOfParticlesInFile = mTotalNumberOfParticles;

    if (mChain.has_variable("ParticleName"))
    {
        mChain.read_variable("ParticleName", particleName, 64);
    }

    if (mChain.has_variable("PDGCode"))
    {
        mChain.read_variable("PDGCode", &mPDGCode);
    }

    // switch to single particle or pairs
    if (mChain.has_variable("X1"))
        InitializeROOTPairs();
    else
        InitializeROOTSingle();

    mInitialized = true;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializeROOTSingle()
{
    mIsPair = false;
    mChain.read_variable("Ekine", &energy);
    mChain.read_variable("X", &x);
    mChain.read_variable("Y", &y);
    mChain.read_variable("Z", &z);
    mChain.read_variable("dX", &dx);
    mChain.read_variable("dY", &dy);
    mChain.read_variable("dZ", &dz);

    if (mChain.has_variable("Weight") and not mIgnoreWeight)
        mChain.read_variable("Weight", &weight);

    if (mTimeIsUsed and mChain.has_variable("Time"))
    {
        if (mChain.get_type_of_variable("Time") == typeid(float))
        {
            mChain.read_variable("Time", &ftime);
            time_type = typeid(float);
        }
        else
        {
            mChain.read_variable("Time", &dtime);
            time_type = typeid(double);
        }
    }
    if (mTimeIsUsed and !mChain.has_variable("Time"))
    {
        GateError("The option 'ignoreTime' is false, but no time was found in the phsp.");
    }
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializeROOTPairs()
{
    mIsPair = true;

    //  E1 E2 X1 Y1 Z1 X2 Y2 Z2 dX1 dY1 dZ1 dX2 dY2 dZ2 t1 t2
    mChain.read_variable("E1", &E1);
    mChain.read_variable("E2", &E2);

    if (mChain.has_variable("t1") and mChain.has_variable("t2") and mTimeIsUsed)
    {
        mChain.read_variable("t1", &t1);
        mChain.read_variable("t2", &t2);
    }
    if (mTimeIsUsed and (!mChain.has_variable("t1") or !mChain.has_variable("t2")))
    {
        GateError("The option 'ignoreTime' is false, but no time t1 and t2 was found in the phsp.");
    }

    mChain.read_variable("X1", &X1);
    mChain.read_variable("Y1", &Y1);
    mChain.read_variable("Z1", &Z1);

    mChain.read_variable("X2", &X2);
    mChain.read_variable("Y2", &Y2);
    mChain.read_variable("Z2", &Z2);

    mChain.read_variable("dX1", &dX1);
    mChain.read_variable("dY1", &dY1);
    mChain.read_variable("dZ1", &dZ1);

    mChain.read_variable("dX2", &dX2);
    mChain.read_variable("dY2", &dY2);
    mChain.read_variable("dZ2", &dZ2);

    // consider one single weight
    if (mChain.has_variable("Weight") and not mIgnoreWeight)
    {
        mChain.read_variable("w1", &w1);
        mChain.read_variable("w2", &w2);
    }
    else
    {
        w1 = w2 = 1.0;
    }
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializePyTorch()
{
    GateMessage("Actor", 1, "GateSourcePhaseSpace InitializePyTorch" << std::endl);

#ifdef GATE_USE_TORCH
    // read the parameters (json file)
    try
    {
        std::ifstream mPTParamFile(mPTJsonFilename);
        mPTParamFile >> mPTParam;
    }
    catch (std::exception &e)
    {
        GateError("GateSourcePhaseSpace: cannot open json file: " << mPTJsonFilename);
    }

    // check if single or pairs
    mIsPair = false; // default is : not a pair
    try
    {
        mIsPair = (mPTParam["is_pairs"] != 0);
    }
    catch (...)
    {
    }

    // check if need to un-normalized or not
    mPTnormalize = true; // default is: apply normalization
    try
    {
        mPTnormalize = (mPTParam["gate_apply_denormalization"] != 0);
    }
    catch (...)
    {
    }

    if (mPTnormalize)
    {
        try
        {
            mPT_x_mean.assign(mPTParam["x_mean"].begin(), mPTParam["x_mean"].end());
            mPT_x_std.assign(mPTParam["x_std"].begin(), mPTParam["x_std"].end());
        }
        catch (...)
        {
            GateError("x_mean and x_std needed in the json file" << mPTJsonFilename);
        }
    }

    // open the .pt file to load the model
    auto filename = listOfPhaseSpaceFile[0];
    GateMessage("Actor", 1, "GateSourcePhaseSpace GAN reading " << filename << " and " << mPTJsonFilename << " is pair ? " << mIsPair << " de-normalize in Gate ? " << mPTnormalize << std::endl);
    try
    {
        mPTmodule = torch::jit::load(filename);
    }
    catch (...)
    {
        GateError("GateSourcePhaseSpace: cannot open the .pt file: " << filename);
    }

    // No CUDA mode yet
    // mPTmodule.to(torch::kCUDA);

    // list of keys
    try
    {
        std::vector<std::string> k = mPTParam["keys"];
        mPT_keys = k;
    }
    catch (std::exception &e)
    {
        std::vector<std::string> k = mPTParam["keys_list"];
        mPT_keys = k;
    }
    if (mPT_keys.size() < 1)
    {
        GateError("GateSourcePhaseSpace: error in json file: keys or keys_list is needed " << mPTJsonFilename);
    }

    // number of dimension for the z GAN input
    mPTz_dim = mPTParam["z_dim"];

    // Create a vector of random inputs
    mPTzer = torch::zeros({mPTBatchSize, mPTz_dim});

    // get particle name
    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    if (mParticleTypeNameGivenByUser == "none")
    {
        GateError("No particle type defined. Use macro setParticleType");
    }
    pParticleDefinition = particleTable->FindParticle(mParticleTypeNameGivenByUser);
    mPTmass = pParticleDefinition->GetPDGMass();
    strcpy(particleName, mParticleTypeNameGivenByUser);

    // set current index to batch size to force compute a first batch of samples
    mPTCurrentIndex = mPTBatchSize;
    // dummy variables
    mCurrentParticleNumberInFile = 1e12;
    mTotalNumberOfParticles = 1e12;
    mNumberOfParticlesInFile = 1e12;

    // verbose
    std::ostringstream oss;
    for (auto k : mPT_keys)
        oss << k << " ";
    GateMessage("Actor", 1, "GateSourcePhaseSpace GAN keys = " << s << std::endl);
    GateMessage("Actor", 1, "GateSourcePhaseSpace GAN z_dim = " << mPTz_dim << " particle = " << particleName << " batch size = " << mPTBatchSize << std::endl);

    // consider single or pair of particle
    if (mIsPair)
        InitializePyTorchPairs();
    else
        InitializePyTorchSingle();

    mInitialized = true;
#endif
}

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializePyTorchPairs()
{
    // allocate batch samples of particles
    for (auto i = 0; i < 2; i++)
    {
        mPTPositions[i].resize(mPTBatchSize);
        mPTDirections[i].resize(mPTBatchSize);
        mPTEnergies[i].resize(mPTBatchSize);
        mPTTimes[i].resize(mPTBatchSize);
    }

    // get index
    mPTPositionIndex[0][0] = get_key_index("X1");
    mPTPositionIndex[0][1] = get_key_index("Y1");
    mPTPositionIndex[0][2] = get_key_index("Z1");

    mPTPositionIndex[1][0] = get_key_index("X2");
    mPTPositionIndex[1][1] = get_key_index("Y2");
    mPTPositionIndex[1][2] = get_key_index("Z2");

    mPTDirectionIndex[0][0] = get_key_index("dX1");
    mPTDirectionIndex[0][1] = get_key_index("dY1");
    mPTDirectionIndex[0][2] = get_key_index("dZ1");

    mPTDirectionIndex[1][0] = get_key_index("dX2");
    mPTDirectionIndex[1][1] = get_key_index("dY2");
    mPTDirectionIndex[1][2] = get_key_index("dZ2");

    mPTEnergiesIndex[0] = get_key_index("E1");
    mPTEnergiesIndex[1] = get_key_index("E2");
    mPTTimeIndex[0] = get_key_index("t1");
    mPTTimeIndex[1] = get_key_index("t2");
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializePyTorchSingle()
{
    // allocate batch samples of particles
    mPTPosition.resize(mPTBatchSize);
    mPTDX.resize(mPTBatchSize);
    mPTDY.resize(mPTBatchSize);
    mPTDZ.resize(mPTBatchSize);
    mPTEnergy.resize(mPTBatchSize);

    mPTPositionXIndex = get_key_index("X");
    mPTPositionYIndex = get_key_index("Y");
    mPTPositionZIndex = get_key_index("Z");
    mPTDirectionXIndex = get_key_index("dX");
    mPTDirectionYIndex = get_key_index("dY");
    mPTDirectionZIndex = get_key_index("dZ");
    mPTEnergyIndex = get_key_index("Ekine");
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateROOTVertex(G4Event * /*aEvent*/)
{
    if (pListOfSelectedEvents.size())
        mChain.read_entrie(pListOfSelectedEvents[mCurrentParticleNumberInFile]);
    else
        mChain.read_entrie(mCurrentParticleNumberInFile);

    G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    G4IonTable *ionTable = G4IonTable::GetIonTable();

    // std::cout
    //     << "mPDGCode " << mPDGCode << " mPDGCodeGivenByUser: " << mPDGCodeGivenByUser << " particleName: " << particleName << " mParticleTypeNameGivenByUser: " << mParticleTypeNameGivenByUser << "\n";
    // particleTable->DumpTable();

    // if PDGCode exists, use this one before particleName
    if (mPDGCode != 0)
    {
        pParticleDefinition = particleTable->FindParticle((G4int)mPDGCode);
        if (pParticleDefinition == 0)
        {
            pParticleDefinition = ionTable->GetIon((G4int)mPDGCode);
        }
        // std::cout << "in here \n";
        // std::cout << "pParticleDefinition: " << pParticleDefinition << "\n";
    }
    else
        pParticleDefinition = particleTable->FindParticle(particleName);
    // if no valid PDGCode or particleName was found, use the user defined ones
    // first PDGCode is checked, if not found or valid, particleName is used
    if (pParticleDefinition == 0)
    {
        if (mPDGCodeGivenByUser != 0)
        {
            pParticleDefinition = particleTable->FindParticle(mPDGCodeGivenByUser);
        }
        if (pParticleDefinition == 0)
        {
            if (mParticleTypeNameGivenByUser != "none")
            {
                pParticleDefinition = particleTable->FindParticle(mParticleTypeNameGivenByUser);
            }
        }
        if (pParticleDefinition == 0)
            GateError("No particle type or PDGCode defined in phase space file or by user.");
    }
    // std::cout << "pParticleDefinition: " << pParticleDefinition << "\n";

    if (mIsPair)
        GenerateROOTVertexPairs();
    else
        GenerateROOTVertexSingle();
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateROOTVertexSingle()
{
    mParticlePosition = G4ThreeVector(x * mm, y * mm, z * mm);

    // parameter not used: double charge = particle_definition->GetPDGCharge();
    double mass = pParticleDefinition->GetPDGMass();

    double dtot = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (energy < 0)
        GateError("Energy < 0 in phase space file!");
    if (energy == 0)
        GateError("Energy = 0 in phase space file!");

    mMomentum = std::sqrt(energy * energy + 2 * energy * mass);

    if (dtot == 0)
        GateError("No momentum defined in phase space file!");

    px = mMomentum * dx / dtot;
    py = mMomentum * dy / dtot;
    pz = mMomentum * dz / dtot;

    mParticleMomentum = G4ThreeVector(px, py, pz);

    if (mTimeIsUsed)
    {
        if (time_type == typeid(double) and dtime > 0)
            mParticleTime = dtime;
        if (time_type == typeid(float) and ftime > 0)
            mParticleTime = ftime;
    }
    else
    {
        mParticleTime = GetTime();
    }
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateROOTVertexPairs()
{
    mParticlePositionPair1 = G4ThreeVector(X1 * mm, Y1 * mm, Z1 * mm);
    mParticlePositionPair2 = G4ThreeVector(X2 * mm, Y2 * mm, Z2 * mm);

    double mass = pParticleDefinition->GetPDGMass();

    double dtot1 = std::sqrt(dX1 * dX1 + dY1 * dY1 + dZ1 * dZ1);
    double dtot2 = std::sqrt(dX2 * dX2 + dY2 * dY2 + dZ2 * dZ2);

    if (E1 <= 0 or E2 <= 0)
    {
        GateError("Energy <0 or =0 in phase space file. Cannot deal with that.");
    }

    auto mMomentum1 = std::sqrt(E1 * E1 + 2 * E1 * mass);
    auto mMomentum2 = std::sqrt(E2 * E2 + 2 * E2 * mass);

    if (dtot1 == 0 or dtot2 == 0)
        GateError("No momentum defined in phase space file!");

    auto ppx = mMomentum1 * dX1 / dtot1;
    auto ppy = mMomentum1 * dY1 / dtot1;
    auto ppz = mMomentum1 * dZ1 / dtot1;
    mParticleMomentumPair1 = G4ThreeVector(ppx, ppy, ppz);

    ppx = mMomentum2 * dX2 / dtot2;
    ppy = mMomentum2 * dY2 / dtot2;
    ppz = mMomentum2 * dZ2 / dtot2;
    mParticleMomentumPair2 = G4ThreeVector(ppx, ppy, ppz);

    if (mTimeIsUsed)
    {
        // nothing, t1 and t2 are read in the file
    }
    else
    {
        t1 = GetTime();
        t2 = GetTime();
    }
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GeneratePyTorchVertex(G4Event *aEvent)
{
    if (mIsPair)
        GeneratePyTorchVertexPairs(aEvent);
    else
        GeneratePyTorchVertexSingle(aEvent);
}

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GeneratePyTorchVertexPairs(G4Event * /*aEvent*/)
{
    if (mPTCurrentIndex >= mPTCurrentBatchSize)
        GenerateBatchSamplesFromPyTorchPairs();

    // Position
    mParticlePositionPair1 = mPTPositions[0][mPTCurrentIndex];
    mParticlePositionPair2 = mPTPositions[1][mPTCurrentIndex];

    // Time
    t1 = mPTTimes[0][mPTCurrentIndex];
    t2 = mPTTimes[1][mPTCurrentIndex];

    // Direction
    for (int i = 0; i < 2; i++)
    {
        dx = mPTDirections[i][mPTCurrentIndex].x();
        dy = mPTDirections[i][mPTCurrentIndex].y();
        dz = mPTDirections[i][mPTCurrentIndex].z();

        // Energy
        energy = mPTEnergies[i][mPTCurrentIndex];
        if (energy <= 0)
        {
            // GAN generated particle may lead to E<0.
            GateWarning("GateSourcePhaseSpace Energy <0 generated by the GAN. Set it to 1e-15");
            energy = 1e-15;
        }

        double dtot = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dtot == 0)
            GateWarning("GateSourcePhaseSpace No momentum defined in GAN generated phase space");

        mMomentum = std::sqrt(energy * energy + 2 * energy * mPTmass);
        px = mMomentum * dx / dtot;
        py = mMomentum * dy / dtot;
        pz = mMomentum * dz / dtot;

        if (i == 0)
            mParticleMomentumPair1 = G4ThreeVector(px, py, pz);
        else
            mParticleMomentumPair2 = G4ThreeVector(px, py, pz);
    }

    // increment
    mPTCurrentIndex++;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GeneratePyTorchVertexSingle(G4Event * /*aEvent*/)
{
    if (mPTCurrentIndex >= mPTCurrentBatchSize)
        GenerateBatchSamplesFromPyTorchSingle();

    // Position
    mParticlePosition = mPTPosition[mPTCurrentIndex];

    // Direction
    dx = mPTDX[mPTCurrentIndex];
    dy = mPTDY[mPTCurrentIndex];
    dz = mPTDZ[mPTCurrentIndex];

    // Energy
    energy = mPTEnergy[mPTCurrentIndex];
    if (energy <= 0)
    {
        // GAN generated particle may lead to E<0.
        GateWarning("Energy <0 generated by the GAN. Set it to 1e-15");
        energy = 1e-15;
    }

    double dtot = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dtot == 0)
        GateError("No momentum defined in GAN generated phase space");

    mMomentum = std::sqrt(energy * energy + 2 * energy * mPTmass);
    px = mMomentum * dx / dtot;
    py = mMomentum * dy / dtot;
    pz = mMomentum * dz / dtot;

    mParticleMomentum = G4ThreeVector(px, py, pz);

    // increment
    mPTCurrentIndex++;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateIAEAVertex(G4Event * /*aEvent*/)
{
    pIAEARecordType->read_particle();

    switch (pIAEARecordType->particle)
    {
    case 1:
        pParticleDefinition = G4Gamma::Gamma();
        break;
    case 2:
        pParticleDefinition = G4Electron::Electron();
        break;
    case 3:
        pParticleDefinition = G4Positron::Positron();
        break;
    case 4:
        pParticleDefinition = G4Neutron::Neutron();
        break;
    case 5:
        pParticleDefinition = G4Proton::Proton();
        break;
    default:
        GateError("Source phase space: particle not available in IAEA phase space format.");
    }
    // particle momentum
    // pc = sqrt(Ek^2 + 2*Ek*m_0*c^2)
    // sqrt( p*cos(Ax)^2 + p*cos(Ay)^2 + p*cos(Az)^2 ) = p
    mMomentum = sqrt(pIAEARecordType->energy * pIAEARecordType->energy +
                     2. * pIAEARecordType->energy * pParticleDefinition->GetPDGMass());

    dx = pIAEARecordType->u;
    dy = pIAEARecordType->v;
    dz = pIAEARecordType->w;

    mParticleMomentum = G4ThreeVector(dx * mMomentum, dy * mMomentum, dz * mMomentum);

    x = pIAEARecordType->x * cm;
    y = pIAEARecordType->y * cm;
    z = pIAEARecordType->z * cm;
    mParticlePosition = G4ThreeVector(x, y, z);

    weight = pIAEARecordType->weight;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
G4int GateSourcePhaseSpace::GeneratePrimaries(G4Event *event)
{
    G4int numVertices = 0;
    double timeSlice = 0.;

    if (mCurrentRunNumber < GateUserActions::GetUserActions()->GetCurrentRun()->GetRunID())
    {
        mCurrentRunNumber = GateUserActions::GetUserActions()->GetCurrentRun()->GetRunID();
        mCurrentUse = 0;
        mLoopFile = 0;
        mCurrentParticleNumberInFile = mStartingParticleId;
        mRequestedNumberOfParticlesPerRun = 0.;
        mLastPartIndex = mCurrentParticleNumber;

        if (GateApplicationMgr::GetInstance()->GetNumberOfPrimariesPerRun())
            mRequestedNumberOfParticlesPerRun = GateApplicationMgr::GetInstance()->GetNumberOfPrimariesPerRun();

        if (GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries())
        {
            timeSlice = GateApplicationMgr::GetInstance()->GetTimeSlice(mCurrentRunNumber);
            mRequestedNumberOfParticlesPerRun =
                GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries() * timeSlice / mTotalSimuTime +
                mResiduRun;
            mResiduRun = mRequestedNumberOfParticlesPerRun - int(mRequestedNumberOfParticlesPerRun);
        }
        mLoop = int(mRequestedNumberOfParticlesPerRun / mTotalNumberOfParticles);

        mAngle = twopi / (mLoop);
    } // Calculate the number of time each particle in phase space will be used

    if (mCurrentUse == 0)
    {
        // mCurrentUse=-1;
        if (mFileType == "IAEAFile")
        {
            if (mCurrentParticleNumberInFile >= mNumberOfParticlesInFile || mCurrentParticleNumberInFile == -1)
            {
                mCurrentParticleNumberInFile = 0;
                if ((int)listOfPhaseSpaceFile.size() <= mLoopFile)
                    mLoopFile = 0;

                mNumberOfParticlesInFile = OpenIAEAFile(G4String(removeExtension(listOfPhaseSpaceFile[mLoopFile])));
                mLoopFile++;
            }
            if (pListOfSelectedEvents.size())
            {
                while (pListOfSelectedEvents[mCurrentUsedParticleInIAEAFiles] > mCurrentParticleInIAEAFiles)
                {
                    if (!mAlreadyLoad)
                        pIAEARecordType->read_particle();

                    mAlreadyLoad = false;
                    mCurrentParticleInIAEAFiles++;
                    mCurrentParticleNumberInFile++;
                    if (mCurrentParticleNumberInFile >= mNumberOfParticlesInFile ||
                        mCurrentParticleNumberInFile == -1)
                    {
                        mCurrentParticleNumberInFile = 0;
                        if ((int)listOfPhaseSpaceFile.size() <= mLoopFile)
                            mLoopFile = 0;
                        mNumberOfParticlesInFile = OpenIAEAFile(
                            G4String(removeExtension(listOfPhaseSpaceFile[mLoopFile])));
                        mLoopFile++;
                    }
                }
            }
            GenerateIAEAVertex(event);
            mAlreadyLoad = true;
            mCurrentParticleNumberInFile++;
            mCurrentParticleInIAEAFiles++;
            mCurrentUsedParticleInIAEAFiles++;
        }
        else if (mFileType == "pytorch")
        {
            GeneratePyTorchVertex(event);
            // Dummy variables
            mCurrentParticleNumberInFile++;
        }
        else
        {
            if (mCurrentParticleNumberInFile >= mNumberOfParticlesInFile)
            {
                mCurrentParticleNumberInFile = 0;
            }
            GenerateROOTVertex(event);
            mCurrentParticleNumberInFile++;
        }
        mResidu = mRequestedNumberOfParticlesPerRun - mTotalNumberOfParticles * mLoop;
    }

    // Generate on or two particles
    if (mIsPair)
        GeneratePrimariesPairs(event);
    else
        GeneratePrimariesSingle(event);

    mCurrentUse++;

    if ((mCurrentParticleNumber - mLastPartIndex) < ((mLoop + 1) * mResidu) && mLoop < mCurrentUse)
        mCurrentUse = 0;
    else if ((mCurrentParticleNumber - mLastPartIndex) >= ((mLoop + 1) * mResidu) &&
             (mLoop - 1) < mCurrentUse)
        mCurrentUse = 0;

    mCurrentParticleNumber++;
    for (auto i = 0; i < event->GetNumberOfPrimaryVertex(); i++)
    {
        auto vertex = event->GetPrimaryVertex(i);
        GateMessage("Beam", 3, "(" << event->GetEventID() << ") " << vertex->GetPrimary()->GetG4code()->GetParticleName() << " pos=" << vertex->GetPosition() << " ene=" << G4BestUnit(vertex->GetPrimary()->GetMomentum().mag(), "Energy") << " momentum=" << vertex->GetPrimary()->GetMomentum() << " weight=" << vertex->GetWeight() << " time=" << G4BestUnit(vertex->GetT0(), "Time") << Gateendl);
    }
    numVertices++;
    return numVertices;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GeneratePrimariesSingle(G4Event *event)
{
    UpdatePositionAndMomentum(mParticlePosition, mParticleMomentum);
    // Timing : relative or absolute ?
    auto t = mParticleTime;
    if (mRelativeTimeFlag)
        t += GetTime();
    GenerateVertex(event, mParticlePosition, mParticleMomentum, t, weight);
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GeneratePrimariesPairs(G4Event *event)
{
    UpdatePositionAndMomentum(mParticlePositionPair1, mParticleMomentumPair1);
    UpdatePositionAndMomentum(mParticlePositionPair2, mParticleMomentumPair2);

    // WARNING : must be double because t1 could be very small
    // (ns) while GetTime could be large (in sec or min)
    double ct1 = t1;
    double ct2 = t2;

    // Timing : relative or absolute ?
    if (mTimeIsUsed and mRelativeTimeFlag)
    {
        ct1 += GetTime();
        ct2 += GetTime();
    }

    /*
    auto app = GateApplicationMgr::GetInstance();
    auto at = app->GetCurrentTime();
    std::cout << "Pairs get Time " << G4BestUnit(GetTime(), "Time") << std::endl;
    std::cout << "Pairs times 1) " << G4BestUnit(t1, "Time") << " " << G4BestUnit(ct1, "Time") << std::endl;
    std::cout << "Pairs times 2) " << G4BestUnit(t2, "Time") << " " << G4BestUnit(ct2, "Time") << std::endl;
    std::cout << "Pairs times 1) " << t1 << " " << ct1 << std::endl;
    std::cout << "Pairs times 2) " << t2 << " " << ct2 << std::endl;
     */

    GenerateVertex(event, mParticlePositionPair1, mParticleMomentumPair1, ct1, weight);
    GenerateVertex(event, mParticlePositionPair2, mParticleMomentumPair2, ct2, weight);

    // Get both vertices
    auto vertex1 = event->GetPrimaryVertex(0);
    auto vertex2 = event->GetPrimaryVertex(1);

    // Set timing
    vertex1->SetT0(ct1);
    vertex2->SetT0(ct2);

    // Set weight
    vertex1->SetWeight(w1);
    vertex2->SetWeight(w2);
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateVertex(G4Event *event, G4ThreeVector &position,
                                          G4ThreeVector &momentum, double time, double w)
{
    pParticle = new G4PrimaryParticle(pParticleDefinition,
                                      momentum.x(),
                                      momentum.y(),
                                      momentum.z());
    pVertex = new G4PrimaryVertex(position, time);
    pVertex->SetWeight(w);
    pVertex->SetPrimary(pParticle);
    event->AddPrimaryVertex(pVertex);
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::UpdatePositionAndMomentum(G4ThreeVector &position, G4ThreeVector &momentum)
{
    G4RotationMatrix rotation;

    // Momentum
    if (GetPositionInWorldFrame())
        momentum = SetReferenceMomentum(momentum);
    // momentum: convert world frame coordinate to local volume coordinate

    if (GetUseRegularSymmetry() && mCurrentUse != 0)
    {
        rotation.rotateZ(mAngle * mCurrentUse);
        momentum = rotation * momentum;
    }
    if (GetUseRandomSymmetry() && mCurrentUse != 0)
    {
        G4double randAngle = G4RandFlat::shoot(twopi);
        rotation.rotateZ(randAngle);
        momentum = rotation * momentum;
    }

    ChangeParticleMomentumRelativeToAttachedVolume(momentum);

    // Position
    // convert world frame coordinate to local volume coordinate
    if (GetPositionInWorldFrame())
        position = SetReferencePosition(position);

    if (GetUseRegularSymmetry() && mCurrentUse != 0)
    {
        position = rotation * position;
    }
    if (GetUseRandomSymmetry() && mCurrentUse != 0)
    {
        position = rotation * position;
    }

    ChangeParticlePositionRelativeToAttachedVolume(position);
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::AddFile(G4String file)
{
    G4String extension = getExtension(file);

    if (listOfPhaseSpaceFile.size() == 0)
    {
        if (extension == "IAEAphsp" || extension == "IAEAheader")
            mFileType = "IAEAFile";
        if (extension == "pt")
            mFileType = "pytorch";
        if (extension == "root")
            mFileType = "root";
        if (extension == "npy")
            mFileType = "root";
    }

    if ((extension == "IAEAphsp" || extension == "IAEAheader"))
    {
        if (mFileType == "IAEAFile")
            listOfPhaseSpaceFile.push_back(file);
        else
            GateError("Cannot mix phase IAEAFile space files with others types");
    }
    if ((mFileType == "pytorch") && (listOfPhaseSpaceFile.size() > 1))
    {
        GateError("Please, use only one pytorch file.");
    }

    if (extension != "IAEAphsp" && extension != "IAEAheader" &&
        extension != "npy" && extension != "root" && extension != "pt")
        GateError("Unknow phase space file extension. Knowns extensions are : "
                  << Gateendl
                  << ".IAEAphsp (or IAEAheader) .root .npy .pt (pytorch) \n");

    listOfPhaseSpaceFile.push_back(file);
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------

void GateSourcePhaseSpace::SetRelativeTimeFlag(bool b)
{
    mRelativeTimeFlag = b;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::SetIgnoreTimeFlag(bool b)
{
    mTimeIsUsed = !b;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
G4ThreeVector GateSourcePhaseSpace::SetReferencePosition(G4ThreeVector coordLocal)
{
    for (int j = mListOfRotation.size() - 1; j >= 0; j--)
    {
        // for(int j = 0; j<mListOfRotation.size();j++){
        const G4ThreeVector &t = mListOfTranslation[j];
        const G4RotationMatrix *r = mListOfRotation[j];
        coordLocal = coordLocal + t;
        if (r)
            coordLocal = (*r) * coordLocal;
    }
    return coordLocal;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
G4ThreeVector GateSourcePhaseSpace::SetReferenceMomentum(G4ThreeVector coordLocal)
{
    for (int j = mListOfRotation.size() - 1; j >= 0; j--)
    {
        const G4RotationMatrix *r = mListOfRotation[j];
        if (r)
            coordLocal = (*r) * coordLocal;
    }
    return coordLocal;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializeTransformation()
{
    GateVVolume *v = mVolume;
    if (v == 0)
        return;
    while (v->GetObjectName() != "world")
    {
        const G4RotationMatrix *r = v->GetPhysicalVolume(0)->GetFrameRotation();
        const G4ThreeVector &t = v->GetPhysicalVolume(0)->GetFrameTranslation();
        mListOfRotation.push_back(r);
        mListOfTranslation.push_back(t);
        // next volume
        v = v->GetParentVolume();
    }
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
G4int GateSourcePhaseSpace::OpenIAEAFile(G4String file)
{
    G4String IAEAFileName = file;
    G4String IAEAHeaderExt = ".IAEAheader";
    G4String IAEAFileExt = ".IAEAphsp";

    if (pIAEAFile)
        fclose(pIAEAFile);
    pIAEAFile = 0;
    free(pIAEAheader);
    free(pIAEARecordType);
    pIAEAheader = 0;
    pIAEARecordType = 0;

    pIAEAFile = open_file(const_cast<char *>(IAEAFileName.c_str()), const_cast<char *>(IAEAFileExt.c_str()),
                          (char *)"rb");
    if (!pIAEAFile)
        GateError("Error file not found: " + IAEAFileName + IAEAFileExt);

    pIAEAheader = (iaea_header_type *)calloc(1, sizeof(iaea_header_type));
    pIAEAheader->fheader = open_file(const_cast<char *>(IAEAFileName.c_str()),
                                     const_cast<char *>(IAEAHeaderExt.c_str()), (char *)"rb");

    if (!pIAEAheader->fheader)
        GateError("Error file not found: " + IAEAFileName + IAEAHeaderExt);
    if (pIAEAheader->read_header())
        GateError("Error reading phase space file header: " + IAEAFileName + IAEAHeaderExt);

    pIAEARecordType = (iaea_record_type *)calloc(1, sizeof(iaea_record_type));
    pIAEARecordType->p_file = pIAEAFile;
    pIAEARecordType->initialize();
    pIAEAheader->get_record_contents(pIAEARecordType);

    return pIAEAheader->nParticles;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateBatchSamplesFromPyTorchPairs()
{
    // the following ifdef prevents to compile this section if GATE_USE_TORCH is not set
#ifdef GATE_USE_TORCH
    // timing
    // auto start = std::chrono::high_resolution_clock::now();

    // nb of particles to generate
    int N = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries();
    if (N != 0 and (N - mCurrentParticleNumberInFile < mPTBatchSize))
    {
        std::cout << "N " << N << " " << mCurrentParticleNumberInFile << std::endl;
        int n = N - mCurrentParticleNumberInFile + 10;
        mPTzer = torch::zeros({n, mPTz_dim});
    }
    GateMessage("Beam", 2, "GAN Phase space, generating " << mPTzer.size(0) << " / " << mPTBatchSize << " particles " << G4endl);

    // Create a vector of random inputs
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor z = torch::randn_like(mPTzer);
    inputs.push_back(z);

    // no CUDA yet
    // inputs.push_back(z.cuda());

    // Execute the model
    // this is the time consuming part
    torch::Tensor output = mPTmodule.forward(inputs).toTensor();

    // Store the results into the vectors
    for (auto a = 0; a < 2; a++)
    {
        for (auto i = 0; i < output.sizes()[0]; ++i)
        {
            const float *v = output[i].data_ptr<float>();
            mPTEnergies[a][i] = get_key_value(v, mPTEnergiesIndex[a], -1);
            mPTPositions[a][i] = G4ThreeVector(get_key_value(v, mPTPositionIndex[a][0], -1),
                                               get_key_value(v, mPTPositionIndex[a][1], -1),
                                               get_key_value(v, mPTPositionIndex[a][2], -1));
            mPTDirections[a][i] = G4ThreeVector(get_key_value(v, mPTDirectionIndex[a][0], -1),
                                                get_key_value(v, mPTDirectionIndex[a][1], -1),
                                                get_key_value(v, mPTDirectionIndex[a][2], -1));
            mPTTimes[a][i] = get_key_value(v, mPTTimeIndex[a], 0.0);
        }
    }

    // debug print
    /*
    for (auto i = 0; i < output.sizes()[0]; ++i) {
        std::cout << i << " E = "
                  << G4BestUnit(mPTEnergies[0][i], "Energy") << " "
                  << G4BestUnit(mPTEnergies[1][i], "Energy")
                  << " t = "
                  << G4BestUnit(mPTTimes[0][i], "Time") << " "
                  << G4BestUnit(mPTTimes[0][i], "Time")
                  << " pos = " << mPTPositions[0][i] << " " << mPTPositions[1][i]
                  << " dir = " << mPTDirections[0][i] << " " << mPTDirections[1][i]
                  << std::endl;
    }
     */

    mPTCurrentBatchSize = output.sizes()[0];
    mPTCurrentIndex = 0;

    /*
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    */

#endif
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateBatchSamplesFromPyTorchSingle()
{
    // the following ifdef prevents to compile this section if GATE_USE_TORCH is not set
#ifdef GATE_USE_TORCH
    // timing
    // auto start = std::chrono::high_resolution_clock::now();

    // nb of particles to generate
    int N = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries();
    if (N != 0 and (N - mCurrentParticleNumberInFile < mPTBatchSize))
    {
        int n = N - mCurrentParticleNumberInFile + 10;
        mPTzer = torch::zeros({n, mPTz_dim});
    }
    GateMessage("Beam", 2, "GAN Phase space, generating " << mPTzer.size(0) << " particles " << G4endl);

    // Check default values
    double def_E = 0.0;
    double def_X = 0.0;
    double def_Y = 0.0;
    double def_Z = 0.0;
    double def_dX = 0.0;
    double def_dY = 0.0;
    double def_dZ = 0.0;
    if (mPTEnergyIndex == -1)
        def_E = mDefaultKeyValues["Ekine"];
    if (mPTPositionXIndex == -1)
        def_X = mDefaultKeyValues["X"];
    if (mPTPositionYIndex == -1)
        def_Y = mDefaultKeyValues["Y"];
    if (mPTPositionZIndex == -1)
        def_Z = mDefaultKeyValues["Z"];
    if (mPTDirectionXIndex == -1)
        def_dX = mDefaultKeyValues["dX"];
    if (mPTDirectionYIndex == -1)
        def_dY = mDefaultKeyValues["dY"];
    if (mPTDirectionZIndex == -1)
        def_dZ = mDefaultKeyValues["dZ"];

    // Create a vector of random inputs
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor z = torch::randn_like(mPTzer);
    inputs.push_back(z);

    // no CUDA yet
    // inputs.push_back(z.cuda());

    // Execute the model
    // this is the time consuming part
    torch::Tensor output = mPTmodule.forward(inputs).toTensor();

    // Store the results into the vectors
    for (auto i = 0; i < output.sizes()[0]; ++i)
    {
        const float *v = output[i].data_ptr<float>();
        mPTEnergy[i] = get_key_value(v, mPTEnergyIndex, def_E);
        mPTPosition[i] = G4ThreeVector(get_key_value(v, mPTPositionXIndex, def_X),
                                       get_key_value(v, mPTPositionYIndex, def_Y),
                                       get_key_value(v, mPTPositionZIndex, def_Z));
        mPTDX[i] = get_key_value(v, mPTDirectionXIndex, def_dX);
        mPTDY[i] = get_key_value(v, mPTDirectionYIndex, def_dY);
        mPTDZ[i] = get_key_value(v, mPTDirectionZIndex, def_dZ);
    }

    mPTCurrentBatchSize = output.sizes()[0];
    mPTCurrentIndex = 0;

    /*
      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    */

#endif
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
int GateSourcePhaseSpace::get_key_index(std::string key)
{
    // find index in the list of keys or return the default value
    auto d = std::find(mPT_keys.begin(), mPT_keys.end(), key);
    auto index = d - mPT_keys.begin();
    if (d == mPT_keys.end())
    {
        index = -1;
        // Check if the value exist in the json
        try
        {
            double v = mPTParam[key];
            mDefaultKeyValues[key] = v;
        }
        catch (std::exception &e)
        {
            GateError("Cannot find the value for key " << key << " in json file");
        }
    }
    return index;
}
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
double GateSourcePhaseSpace::get_key_value(const float *v, int i, double def)
{
    if (i < 0)
        return def;
    if (mPTnormalize)
        return (v[i] * this->mPT_x_std[i]) + this->mPT_x_mean[i];
    else
        return (double)(v[i]);
}
// ----------------------------------------------------------------------------------
