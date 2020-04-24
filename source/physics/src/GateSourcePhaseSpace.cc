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

typedef unsigned int uint;

// ----------------------------------------------------------------------------------
GateSourcePhaseSpace::GateSourcePhaseSpace(G4String name ):
  GateVSource( name )
{
  mCurrentParticleNumber=0;
  mNumberOfParticlesInFile=0;
  mTotalNumberOfParticles=0;
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
  mAngle=0.;
  mPositionInWorldFrame = false;
  mRequestedNumberOfParticlesPerRun = 0;
  mInitialized  = false;
  m_sourceMessenger = new GateSourcePhaseSpaceMessenger(this);
  mFileType = "";
  mParticleTime = 0. ;//->GetTime();
  pIAEAFile = 0;
  pIAEARecordType = 0;
  pIAEAheader = 0;
  pParticleDefinition = 0;
  pParticle = 0;
  pVertex = 0;
  mMomentum = 0.;
  mParticlePosition = G4ThreeVector();
  mParticleMomentum = G4ThreeVector();
  mParticlePosition2 = G4ThreeVector();
  mParticleMomentum2 = G4ThreeVector();
  x = y = z = dx = dy = dz = px = py = pz = energy  = 0.;
  dtime= -1.;
  ftime= -1.;
  weight = 1.;
  strcpy(particleName, "");
  mPTBatchSize = 1e5;
  mPTCurrentIndex = mPTBatchSize;
  mTotalSimuTime = 0.;
  mAlreadyLoad = false;
  mRmax=0;
  mSphereRadius = -1;
  mCurrentParticleInIAEAFiles = 0;
  mCurrentUsedParticleInIAEAFiles = 0;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
GateSourcePhaseSpace::~GateSourcePhaseSpace()
{
  listOfPhaseSpaceFile.clear();
  //delete translation/rotation vectors

  if (pIAEAFile) fclose(pIAEAFile);
  pIAEAFile = 0;
  free(pIAEAheader);
  free(pIAEARecordType);
  pIAEAheader = 0;
  pIAEARecordType = 0;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::Initialize()
{
  InitializeTransformation();
  mTotalSimuTime = GateApplicationMgr::GetInstance()->GetTimeStop() - GateApplicationMgr::GetInstance()->GetTimeStart();

  if (mFileType == "IAEAFile") {
    int totalEvent = 0;
    int totalEventInFile = 0;
    mCurrentParticleNumberInFile = -1;
    G4String IAEAFileName  = " ";
    for(uint j=0;j<listOfPhaseSpaceFile.size();j++) {
      IAEAFileName = G4String(removeExtension(listOfPhaseSpaceFile[j]));
      totalEventInFile = OpenIAEAFile(IAEAFileName);
      mTotalNumberOfParticles += totalEventInFile;

      if (mRmax>0){
        for(int j=0 ; j<totalEventInFile ; j++) {
          pIAEARecordType->read_particle();
          if (std::abs(pIAEARecordType->x*cm)<mRmax && std::abs(pIAEARecordType->y*cm)<mRmax) {
            pListOfSelectedEvents.push_back(totalEvent);
            // G4cout<<" --> OK  "<<totalEvent<< Gateendl;
          }
          totalEvent++;
        }
      }
    }
    if (mRmax>0) mTotalNumberOfParticles = pListOfSelectedEvents.size();
  }

  else if (mFileType == "pytorch") {
    InitializePyTorch();
  }
  else {
    for(auto file: listOfPhaseSpaceFile) {
      GateMessage("Beam", 1, "Phase Space Source. Read file " << file << Gateendl);
      auto extension = getExtension(file);
      mChain.add_file(file, extension);
    }
    mChain.set_tree_name("PhaseSpace");
    mChain.read_header();

    mTotalNumberOfParticles = mChain.nb_elements();
    mNumberOfParticlesInFile = mTotalNumberOfParticles;

    if (mChain.has_variable("ParticleName")) {
      mChain.read_variable("ParticleName",particleName, 64);
    }
    mChain.read_variable("Ekine", &energy);

    mChain.read_variable("X",&x);
    mChain.read_variable("Y",&y);
    mChain.read_variable("Z",&z);
    mChain.read_variable("dX",&dx);
    mChain.read_variable("dY",&dy);
    mChain.read_variable("dZ",&dz);

    if(mChain.has_variable("Weight") and not mIgnoreWeight)
      mChain.read_variable("Weight", &weight);

    if(mChain.has_variable("Time")) {
      if(mChain.get_type_of_variable("Time") == typeid(float)) {
        mChain.read_variable("Time",&ftime);
        time_type = typeid(float);
      }
      else {
        mChain.read_variable("Time",&dtime);
        time_type = typeid(double);
      }
    }

    if (mRmax>0){
      for(int i = 0; i < mTotalNumberOfParticles;i++) {
        mChain.read_entrie(i);
        if (std::abs(x)<mRmax && std::abs(y)<mRmax) {
          pListOfSelectedEvents.push_back(i);
        }
      }
      mTotalNumberOfParticles = pListOfSelectedEvents.size();
      mNumberOfParticlesInFile = mTotalNumberOfParticles;
    }
  }

  mInitialized  = true;

  if (mUseNbOfParticleAsIntensity)
    SetIntensity(mNumberOfParticlesInFile);

  GateMessage("Beam", 1, "Phase Space Source. Total nb of particles in PhS "
              << mNumberOfParticlesInFile << Gateendl);
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateROOTVertex( G4Event* /*aEvent*/ )
{
  if (pListOfSelectedEvents.size()) mChain.read_entrie(pListOfSelectedEvents[mCurrentParticleNumberInFile]);
  else mChain.read_entrie(mCurrentParticleNumberInFile);

  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  pParticleDefinition = particleTable->FindParticle(particleName);

  if (pParticleDefinition==0) {
    if (mParticleTypeNameGivenByUser != "none") {
      pParticleDefinition = particleTable->FindParticle(mParticleTypeNameGivenByUser);
    }
    if (pParticleDefinition==0) GateError("No particle type defined in phase space file.");
  }

  mParticlePosition = G4ThreeVector(x*mm,y*mm,z*mm);

  //parameter not used: double charge = particle_definition->GetPDGCharge();
  double mass =  pParticleDefinition->GetPDGMass();

  double dtot = std::sqrt(dx*dx + dy*dy + dz*dz);

  if (energy<0) GateError("Energy < 0 in phase space file!");
  if (energy==0) GateError("Energy = 0 in phase space file!");

  mMomentum = std::sqrt(energy*energy+2*energy*mass);

  if (dtot==0) GateError("No momentum defined in phase space file!");
  //if (dtot>1) GateError("Sum of square normalized directions should be equal to 1");

  px = mMomentum*dx/dtot ;
  py = mMomentum*dy/dtot ;
  pz = mMomentum*dz/dtot ;

  mParticleMomentum = G4ThreeVector(px,py,pz);

  if (time_type == typeid(double) and dtime>0) mParticleTime = dtime;
  if (time_type == typeid(float) and ftime>0) mParticleTime = ftime;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GeneratePyTorchVertex( G4Event* /*aEvent*/ )
{
  if (mPTCurrentIndex >= mPTBatchSize) GenerateBatchSamplesFromPyTorch();

  // Position
  mParticlePosition = mPTPosition[mPTCurrentIndex];

  // Direction
  dx = mPTDX[mPTCurrentIndex];
  dy = mPTDY[mPTCurrentIndex];
  dz = mPTDZ[mPTCurrentIndex];

  // Energy
  energy = mPTEnergy[mPTCurrentIndex];
  if (energy<=0) {
    // GAN generated particle may lead to E<0.
    energy = 1e-15;
  }

  double dtot = std::sqrt(dx*dx + dy*dy + dz*dz);
  if (dtot==0) GateError("No momentum defined in GAN generated phase space");

  mMomentum = std::sqrt(energy*energy+2*energy*mPTmass);
  px = mMomentum*dx/dtot ;
  py = mMomentum*dy/dtot ;
  pz = mMomentum*dz/dtot ;

  mParticleMomentum = G4ThreeVector(px,py,pz);

  // increment
  mPTCurrentIndex++;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateIAEAVertex( G4Event* /*aEvent*/ )
{
  pIAEARecordType->read_particle();

  switch( pIAEARecordType->particle ){
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
    GateError("Source phase space: particle not available in IAEA phase space format." );
  }
  // particle momentum
  // pc = sqrt(Ek^2 + 2*Ek*m_0*c^2)
  // sqrt( p*cos(Ax)^2 + p*cos(Ay)^2 + p*cos(Az)^2 ) = p
  mMomentum = sqrt( pIAEARecordType->energy*pIAEARecordType->energy + 2.*pIAEARecordType->energy*pParticleDefinition->GetPDGMass() );

  dx = pIAEARecordType->u;
  dy = pIAEARecordType->v;
  dz = pIAEARecordType->w;

  mParticleMomentum = G4ThreeVector(dx*mMomentum, dy*mMomentum, dz*mMomentum);

  x = pIAEARecordType->x*cm;
  y = pIAEARecordType->y*cm;
  z = pIAEARecordType->z*cm;
  mParticlePosition = G4ThreeVector(x,y,z);

  weight = pIAEARecordType->weight;

  //if (mCurrentParticleNumber>=mNumberOfParticlesInFile) mCurrentParticleNumber=0;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
G4int GateSourcePhaseSpace::GeneratePrimaries( G4Event* event )
{
  G4int numVertices = 0;
  double timeSlice = 0.;
  G4RotationMatrix rotation;

  if (mCurrentRunNumber<GateUserActions::GetUserActions()->GetCurrentRun()->GetRunID()) {
    mCurrentRunNumber=GateUserActions::GetUserActions()->GetCurrentRun()->GetRunID();
    mCurrentUse=0;
    mLoopFile=0;
    mCurrentParticleNumberInFile= mStartingParticleId;
    mRequestedNumberOfParticlesPerRun = 0.;
    mLastPartIndex = mCurrentParticleNumber;

    if (GateApplicationMgr::GetInstance()->GetNumberOfPrimariesPerRun())
      mRequestedNumberOfParticlesPerRun = GateApplicationMgr::GetInstance()->GetNumberOfPrimariesPerRun();

    if (GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries()) {
      timeSlice = GateApplicationMgr::GetInstance()->GetTimeSlice(mCurrentRunNumber);
      mRequestedNumberOfParticlesPerRun = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries()*timeSlice/mTotalSimuTime + mResiduRun;
      mResiduRun = mRequestedNumberOfParticlesPerRun - int(mRequestedNumberOfParticlesPerRun);
    }
    mLoop = int(mRequestedNumberOfParticlesPerRun/mTotalNumberOfParticles)  ;

    mAngle = twopi/(mLoop);
  }//Calculate the number of time each particle in phase space will be used


  if (mCurrentUse==0) {
    //mCurrentUse=-1;
    if (mFileType == "IAEAFile"){
      if (mCurrentParticleNumberInFile>=mNumberOfParticlesInFile || mCurrentParticleNumberInFile == -1){
        mCurrentParticleNumberInFile=0;
        if ((int)listOfPhaseSpaceFile.size()<=mLoopFile) mLoopFile=0;

        mNumberOfParticlesInFile = OpenIAEAFile(G4String(removeExtension(listOfPhaseSpaceFile[mLoopFile])));
        mLoopFile++;
      }
      if (pListOfSelectedEvents.size())
        {
          while(pListOfSelectedEvents[mCurrentUsedParticleInIAEAFiles]>mCurrentParticleInIAEAFiles ){
            if (!mAlreadyLoad) pIAEARecordType->read_particle();

            mAlreadyLoad = false;
            mCurrentParticleInIAEAFiles++;
            mCurrentParticleNumberInFile++;
            if (mCurrentParticleNumberInFile>=mNumberOfParticlesInFile || mCurrentParticleNumberInFile == -1){
              mCurrentParticleNumberInFile=0;
              if ((int)listOfPhaseSpaceFile.size()<=mLoopFile) mLoopFile=0;
              mNumberOfParticlesInFile = OpenIAEAFile(G4String(removeExtension(listOfPhaseSpaceFile[mLoopFile])));
              mLoopFile++;
            }

          }
        }
      GenerateIAEAVertex( event );
      mAlreadyLoad = true;
      mCurrentParticleNumberInFile++;
      mCurrentParticleInIAEAFiles++;
      mCurrentUsedParticleInIAEAFiles++;
    }
    else if (mFileType == "pytorch") {
      GeneratePyTorchVertex( event );
      // Dummy variables
      mCurrentParticleNumberInFile++;
    }
    else {
      if (mCurrentParticleNumberInFile>=mNumberOfParticlesInFile) {mCurrentParticleNumberInFile=0;}
      GenerateROOTVertex( event );
      mCurrentParticleNumberInFile++;
    }
    mResidu = mRequestedNumberOfParticlesPerRun-mTotalNumberOfParticles*mLoop;
  }

  mParticleMomentum2 = mParticleMomentum;
  mParticlePosition2 = mParticlePosition;

  if (GetPositionInWorldFrame())
    mParticleMomentum2 = SetReferenceMomentum(mParticleMomentum2);
  //momentum: convert world frame coordinate to local volume coordinate

  if (GetUseRegularSymmetry() && mCurrentUse!=0) {
    rotation.rotateZ(mAngle*mCurrentUse);
    mParticleMomentum2 =  rotation*mParticleMomentum2;
  }
  if (GetUseRandomSymmetry() && mCurrentUse!=0) {
    G4double randAngle = G4RandFlat::shoot(twopi);
    rotation.rotateZ(randAngle);
    mParticleMomentum2 =  rotation*mParticleMomentum2;
  }

  ChangeParticleMomentumRelativeToAttachedVolume(mParticleMomentum2);

  pParticle = new G4PrimaryParticle(pParticleDefinition, mParticleMomentum2.x(), mParticleMomentum2.y(), mParticleMomentum2.z());

  /*particle->SetMass( mass );
    particle->SetCharge( charge );
    particle->SetPolarization( particle_polarization.x(),
    particle_polarization.y(),
    particle_polarization.z());*/

  if (GetPositionInWorldFrame())
    mParticlePosition2 = SetReferencePosition(mParticlePosition2);
  //particle: convert world frame coordinate to local volume coordinate

  if (GetUseRegularSymmetry() && mCurrentUse!=0) { mParticlePosition2  =  rotation*mParticlePosition2; }
  if (GetUseRandomSymmetry() && mCurrentUse!=0) {  mParticlePosition2 =  rotation*mParticlePosition2; }

  ChangeParticlePositionRelativeToAttachedVolume(mParticlePosition2);

  // ----------------------------------------------
  // change position according to a max radius (experimental)
  if (mSphereRadius > 0) {
    //G4cout << "mParticlePosition2 " << mParticlePosition2 << std::endl;
    G4ThreeVector mSphereCenter(0,0,0);
    //G4cout << "Sphere center/radius" << mSphereCenter << " " << mSphereRadius << std::endl;
    G4ThreeVector l = mParticleMomentum2/mParticleMomentum2.mag();
    G4ThreeVector & o = mParticlePosition2;
    G4ThreeVector & c = mSphereCenter;
    double & r = mSphereRadius;
    // G4cout << "  l " << l << std::endl;
    // G4cout << "  o " << o << std::endl;
    // G4cout << "  c " << c << std::endl;
    // G4cout << "  r " << r << std::endl;
    // G4cout << "  norm " << l.mag2() << std::endl;
    G4ThreeVector diff = (o-c);
    double a = -(l.dot(diff));
    double b = a*a - (diff.mag2()-r*r);
    // G4cout << "    diff " << diff << std::endl;
    // G4cout << "    a " << a << std::endl;
    // G4cout << "    a² " << a*a << std::endl;
    // G4cout << "    diff² " << diff.mag2() << std::endl;
    // G4cout << "    r² " << r*r << std::endl;
    // G4cout << "    b " << b << std::endl;

    if (b>0) {
      double c = sqrt(b);
      double d1 = a + c;
      double d2 = a - c;
      // G4cout << "    d1 " << d1 << std::endl;
      // G4cout << "    d2 " << d2 << std::endl;
      if (d1<0 && d2<0) {
        double d = d2;
        if (d1 > d2) d = d1;
        mParticlePosition2 = mParticlePosition2+d*l;
      }
      else {
        // Ignore
        GateMessage("Beam", 2, "WARNING particle is already inside the sphere " << d1 << " " << d2 << std::endl);
      }
    }
  }
  // ----------------------------------------------


  //parameter not used: particle_time
  pVertex = new G4PrimaryVertex(mParticlePosition2, mParticleTime);
  pVertex->SetWeight(weight);
  pVertex->SetPrimary(pParticle);

  event->AddPrimaryVertex(pVertex);

  mCurrentUse++;

  if ((mCurrentParticleNumber-mLastPartIndex) < ((mLoop+1)*mResidu)  && mLoop<mCurrentUse ) mCurrentUse=0;
  else if ((mCurrentParticleNumber-mLastPartIndex) >= ((mLoop+1)*mResidu) && (mLoop-1)<mCurrentUse ) mCurrentUse=0;

  mCurrentParticleNumber++;

  GateMessage("Beam", 3, "(" << event->GetEventID() << ") "
              << pVertex->GetPrimary()->GetG4code()->GetParticleName()
              << " pos=" << pVertex->GetPosition()
              << " ene=" << G4BestUnit(pVertex->GetPrimary()->GetMomentum().mag(), "Energy")
              << " momentum=" << pVertex->GetPrimary()->GetMomentum()
              << " weight=" << pVertex->GetWeight()
              << " time=" << pVertex->GetT0()
              << Gateendl);

  numVertices++;
  return numVertices;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::AddFile(G4String file)
{
  G4String extension = getExtension(file);

  if (listOfPhaseSpaceFile.size()==0) {
    if (extension == "IAEAphsp" || extension == "IAEAheader" )
      mFileType = "IAEAFile";
    if (extension == "pt")
      mFileType = "pytorch";
  }

  if ((extension == "IAEAphsp" || extension == "IAEAheader")) {
    if(mFileType == "IAEAFile")
      listOfPhaseSpaceFile.push_back(file);
    else
      GateError( "Cannot mix phase IAEAFile space files with others types");
  }
  if ((mFileType == "pytorch") && (listOfPhaseSpaceFile.size() > 1)) {
    GateError( "Please, use only one pytorch file.");
  }

  G4cout << "GateSourcePhaseSpace::AddFile Add " << file << G4endl;

  if(extension != "IAEAphsp" && extension != "IAEAheader" && 
    extension != "npy" && extension != "root" && extension != "pt")
    GateError( "Unknow phase space file extension. Knowns extensions are : "
               << Gateendl
               << ".IAEAphsp (or IAEAheader) .root .npy .pt (pytorch) \n");

  listOfPhaseSpaceFile.push_back(file);
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
G4ThreeVector GateSourcePhaseSpace::SetReferencePosition(G4ThreeVector coordLocal)
{
  for(int j = mListOfRotation.size()-1;j>=0;j--) {
    //for(int j = 0; j<mListOfRotation.size();j++){
    const G4ThreeVector & t = mListOfTranslation[j];
    const G4RotationMatrix * r =  mListOfRotation[j];

    coordLocal = coordLocal+t;

    if (r) coordLocal = (*r)*coordLocal;
  }

  return coordLocal;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
G4ThreeVector GateSourcePhaseSpace::SetReferenceMomentum(G4ThreeVector coordLocal)
{
  for(int j = mListOfRotation.size()-1;j>=0;j--){
    const G4RotationMatrix * r =  mListOfRotation[j];

    if (r) coordLocal = (*r)*coordLocal;
  }
  return coordLocal;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializeTransformation()
{
  GateVVolume * v = mVolume;
  if (v==0) return;
  while (v->GetObjectName() != "world") {
    const G4RotationMatrix * r = v->GetPhysicalVolume(0)->GetFrameRotation();
    const G4ThreeVector & t = v->GetPhysicalVolume(0)->GetFrameTranslation();
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
  G4String IAEAFileName  = file;
  G4String IAEAHeaderExt = ".IAEAheader";
  G4String IAEAFileExt   = ".IAEAphsp";

  if (pIAEAFile) fclose(pIAEAFile);
  pIAEAFile = 0;
  free(pIAEAheader);
  free(pIAEARecordType);
  pIAEAheader = 0;
  pIAEARecordType = 0;

  pIAEAFile = open_file(const_cast<char*>(IAEAFileName.c_str()), const_cast<char*>(IAEAFileExt.c_str()),(char*)"rb");
  if (!pIAEAFile) GateError("Error file not found: " + IAEAFileName + IAEAFileExt);

  pIAEAheader = (iaea_header_type *) calloc(1, sizeof(iaea_header_type));
  pIAEAheader->fheader = open_file(const_cast<char*>(IAEAFileName.c_str()), const_cast<char*>(IAEAHeaderExt.c_str()), (char*)"rb");

  if (!pIAEAheader->fheader) GateError("Error file not found: " + IAEAFileName + IAEAHeaderExt);
  if ( pIAEAheader->read_header() ) GateError("Error reading phase space file header: " + IAEAFileName + IAEAHeaderExt);

  pIAEARecordType= (iaea_record_type *) calloc(1, sizeof(iaea_record_type));
  pIAEARecordType->p_file = pIAEAFile;
  pIAEARecordType->initialize();
  pIAEAheader->get_record_contents(pIAEARecordType);

  return pIAEAheader->nParticles;
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::InitializePyTorch()
{
  GateMessage("Actor", 1, "GateSourcePhaseSpace InitializePyTorch" << std::endl);

  // allocate batch samples of particles
  mPTPosition.resize(mPTBatchSize);
  mPTDX.resize(mPTBatchSize);
  mPTDY.resize(mPTBatchSize);
  mPTDZ.resize(mPTBatchSize);
  mPTEnergy.resize(mPTBatchSize);

  // set current index to batch size to force compute a first batch of samples
  mPTCurrentIndex = mPTBatchSize;

  // get particle name
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  if (mParticleTypeNameGivenByUser == "none") {
    GateError("No particle type defined. Use macro setParticleType");
  }
  pParticleDefinition = particleTable->FindParticle(mParticleTypeNameGivenByUser);
  mPTmass =  pParticleDefinition->GetPDGMass();

  // dummy variable
  mCurrentParticleNumberInFile = 1e12;
  mTotalNumberOfParticles = 1e12;
  mNumberOfParticlesInFile = 1e12;

#ifdef GATE_USE_TORCH

  // load the model
  auto filename = listOfPhaseSpaceFile[0];
  GateMessage("Actor", 1, "GateSourcePhaseSpace reading " << filename << std::endl);
  try {
  mPTmodule = torch::jit::load(filename);
}
  // check
  catch (...) {
    GateError("GateSourcePhaseSpace: cannot open the .pt file: " << filename);
  }

  // No CUDA mode yet
  // mPTmodule.to(torch::kCUDA);

  // read json file
  nlohmann::json nnDict;
  try {
    std::ifstream nnDictFile(mPTJsonFilename);
    nnDictFile >> nnDict;
  } catch(std::exception & e) {
    GateError("GateSourcePhaseSpace: cannot open json file: " << mPTJsonFilename);
  }

  // un normalize
  std::vector<double> x_mean = nnDict["x_mean"];
  std::vector<double> x_std = nnDict["x_std"];
  mPTx_mean = x_mean;
  mPTx_std = x_std;

  // list of keys
  std::vector<std::string> keys = nnDict["keys"];
  mPTz_dim = nnDict["z_dim"];
  auto get_index = [&keys, &nnDict, this](std::string s) {
                     auto d = std::find(keys.begin(), keys.end(), s);
                     auto index = d-keys.begin();
                     if (d == keys.end()) {
                       index = -1;
                       // Check if the value exist in the json
                       try {
                         double v = nnDict[s];
                         this->mDefaultKeyValues[s] = v;
                         std::cout << " v = " << v << std::endl;
                       } catch(std::exception & e) {
                         GateError("Cannot find the value for key " << s << " in json file");
                       }
                     }
                     // else this->mPTz_dim++;
                     std::cout << "index for " << s << " = " << index << std::endl;
                     return index;
                   };

  mPTPositionXIndex = get_index("X");
  mPTPositionYIndex = get_index("Y");
  mPTPositionZIndex = get_index("Z");
  mPTDirectionXIndex = get_index("dX");
  mPTDirectionYIndex = get_index("dY");
  mPTDirectionZIndex = get_index("dZ");
  mPTEnergyIndex = get_index("Ekine");

  // Create a vector of random inputs
  mPTzer = torch::zeros({mPTBatchSize, mPTz_dim});

  std::cout << "index : " << mPTPositionXIndex << " " << mPTPositionYIndex << " " << mPTPositionZIndex << std::endl;
  std::cout << "index : " << mPTDirectionXIndex << " " << mPTDirectionYIndex << " " << mPTDirectionZIndex << std::endl;
  std::cout << "index E : " << mPTEnergyIndex << std::endl;

  std::cout << "mean " << mPTx_mean << std::endl;
  std::cout << "std  " << mPTx_std << std::endl;
  std::cout << "Zdim  " << mPTz_dim << std::endl;  

  strcpy(particleName, mParticleTypeNameGivenByUser);
#endif
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
void GateSourcePhaseSpace::GenerateBatchSamplesFromPyTorch()
{
  // the following ifdef prevents to compile this section if GATE_USE_TORCH is not set
#ifdef GATE_USE_TORCH

  // timing
  //auto start = std::chrono::high_resolution_clock::now();

  // nb of particles to generate
  int N = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries();
  if (N!= 0 and (N-mCurrentParticleNumberInFile < mPTBatchSize)) {
    std::cout << "N " << N << " " << mCurrentParticleNumberInFile << std::endl;
    int n = N-mCurrentParticleNumberInFile + 10;
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
  if (mPTEnergyIndex == -1) def_E = mDefaultKeyValues["Ekine"];
  if (mPTPositionXIndex == -1) def_X = mDefaultKeyValues["X"];
  if (mPTPositionYIndex == -1) def_Y = mDefaultKeyValues["Y"];
  if (mPTPositionZIndex == -1) def_Z = mDefaultKeyValues["Z"];
  if (mPTDirectionXIndex == -1) def_dX = mDefaultKeyValues["dX"];
  if (mPTDirectionYIndex == -1) def_dY = mDefaultKeyValues["dY"];
  if (mPTDirectionZIndex == -1) def_dZ = mDefaultKeyValues["dZ"];

  // Create a vector of random inputs
  std::vector<torch::jit::IValue> inputs;
  torch::Tensor z = torch::randn_like(mPTzer);
  inputs.push_back(z);

  // no CUDA yet
  // inputs.push_back(z.cuda());

  // Execute the model
  // this is the time consuming part
  torch::Tensor output = mPTmodule.forward(inputs).toTensor();

  // un normalize
  std::vector<double> x_mean = mPTx_mean;
  std::vector<double> x_std = mPTx_std;
  auto u = [&x_mean, &x_std](const float * v, int i, double def) {
             if (i == -1) return def;
             return (v[i]*x_std[i])+x_mean[i];
           };

  // Store the results into the vectors
  for (auto i=0; i < output.sizes()[0]; ++i) {
    const float * v = output[i].data_ptr<float>();
    mPTEnergy[i] = u(v, mPTEnergyIndex, def_E);
    mPTPosition[i] = G4ThreeVector(u(v, mPTPositionXIndex, def_X),
                                   u(v, mPTPositionYIndex, def_Y),
                                   u(v, mPTPositionZIndex, def_Z));
    mPTDX[i] = u(v,mPTDirectionXIndex, def_dX);
    mPTDY[i] = u(v,mPTDirectionYIndex, def_dY);
    mPTDZ[i] = u(v,mPTDirectionZIndex, def_dZ);
  }

  mPTCurrentIndex = 0;

  /*
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  */

#endif
}
// ----------------------------------------------------------------------------------
