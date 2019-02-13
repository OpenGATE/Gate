/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

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

typedef unsigned int uint;

// ----------------------------------------------------------------------------------
GateSourcePhaseSpace::GateSourcePhaseSpace(G4String name ):GateVSource( name )
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

  mUseRegularSymmetry = false;
  mUseRandomSymmetry = false;
  mAngle=0.;
  mPositionInWorldFrame = false;

  mRequestedNumberOfParticlesPerRun = 0;

  mInitialized  = false;
  m_sourceMessenger = new GateSourcePhaseSpaceMessenger( this );

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
  time_type = EDataType::kOther_t;
  weight = 1.;
  strcpy(particleName, "");

  mTotalSimuTime = 0.;
  mAlreadyLoad = false;
  mRmax=0;
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
  // GateMessage("Beam", 1, "Phase Space Source - Initialisation\n");

  InitializeTransformation();
  mTotalSimuTime = GateApplicationMgr::GetInstance()->GetTimeStop() - GateApplicationMgr::GetInstance()->GetTimeStart();

  if (mFileType == "rootFile"){
    T = new TChain("PhaseSpace");  //creates a chain to process a Tree called "T"

    for(unsigned int i=0;i<listOfPhaseSpaceFile.size();i++) {
      GateMessage("Beam", 1, "Phase Space Source. Read file " << listOfPhaseSpaceFile[i] << Gateendl);
      T->Add(listOfPhaseSpaceFile[i]);
    }

    mTotalNumberOfParticles = T->GetEntries();
    mNumberOfParticlesInFile = mTotalNumberOfParticles;

    if (T->GetListOfBranches()->FindObject("ParticleName")) {
      T->SetBranchAddress("ParticleName",&particleName);
    }
    T->SetBranchAddress("Ekine",&energy);
    T->SetBranchAddress("X",&x);
    T->SetBranchAddress("Y",&y);
    T->SetBranchAddress("Z",&z);
    T->SetBranchAddress("dX",&dx);
    T->SetBranchAddress("dY",&dy);
    T->SetBranchAddress("dZ",&dz);
    if (T->GetListOfBranches()->FindObject("Weight")) {
      T->SetBranchAddress("Weight",&weight);
    }
    auto tob = T->GetListOfBranches()->FindObject("Time");
    if (tob) {
      auto tt = dynamic_cast<TBranch*>(tob);
      TClass * expectedClass;
      tt-> GetExpectedType(expectedClass, time_type);
      if (time_type == EDataType::kDouble_t)
        T->SetBranchAddress("Time",&dtime);
      else
        T->SetBranchAddress("Time",&ftime);
    }

    if (mRmax>0){
      for(int i = 0; i < mTotalNumberOfParticles;i++){
        T->GetEntry(i);
        if (std::abs(x)<mRmax && std::abs(y)<mRmax)
          {
            pListOfSelectedEvents.push_back(i);
          }
      }
      mTotalNumberOfParticles = pListOfSelectedEvents.size();
      mNumberOfParticlesInFile = mTotalNumberOfParticles;
    }
  }

  if (mFileType == "IAEAFile"){
    int totalEvent = 0;
    int totalEventInFile = 0;
    mCurrentParticleNumberInFile = -1;
    G4String IAEAFileName  = " ";
    for(uint j=0;j<listOfPhaseSpaceFile.size();j++){
      IAEAFileName = G4String(removeExtension(listOfPhaseSpaceFile[j]));
      totalEventInFile = OpenIAEAFile(IAEAFileName);
      mTotalNumberOfParticles += totalEventInFile;

      if (mRmax>0){
        for(int j=0 ; j<totalEventInFile ; j++)
          {
            pIAEARecordType->read_particle();
            if ( std::abs(pIAEARecordType->x*cm)<mRmax && std::abs(pIAEARecordType->y*cm)<mRmax )  {pListOfSelectedEvents.push_back(totalEvent);G4cout<<" --> OK  "<<totalEvent<< Gateendl;}
            totalEvent++;
          }
      }
    }
    if (mRmax>0) mTotalNumberOfParticles = pListOfSelectedEvents.size();
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
  if (pListOfSelectedEvents.size()) T->GetEntry(pListOfSelectedEvents[mCurrentParticleNumberInFile]);
  else T->GetEntry(mCurrentParticleNumberInFile);

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

  if (time_type == EDataType::kDouble_t and dtime>0) mParticleTime = dtime;
  if (time_type == EDataType::kFloat_t and ftime>0) mParticleTime = ftime;
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
  //GateMessage("Beam", 2, "Generating particle " << event->GetEventID() << Gateendl);
  //GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << Gateendl);

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


    if (GateApplicationMgr::GetInstance()->GetNumberOfPrimariesPerRun()) mRequestedNumberOfParticlesPerRun = GateApplicationMgr::GetInstance()->GetNumberOfPrimariesPerRun();
    if (GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries()) {
      timeSlice = GateApplicationMgr::GetInstance()->GetTimeSlice(mCurrentRunNumber);

      mRequestedNumberOfParticlesPerRun = GateApplicationMgr::GetInstance()->GetTotalNumberOfPrimaries()*timeSlice/mTotalSimuTime + mResiduRun;
      mResiduRun = mRequestedNumberOfParticlesPerRun - int(mRequestedNumberOfParticlesPerRun);
    }
    mLoop = int(mRequestedNumberOfParticlesPerRun/mTotalNumberOfParticles)  ;

    mAngle = twopi/(mLoop);
  }//Calculate the number of time each particle in phase space will be used


  if (mCurrentUse==0){
    //mCurrentUse=-1;
    if (mFileType == "rootFile") {
      //if (mCurrentParticleNumber>=mTotalNumberOfParticles) {mCurrentParticleNumberInFile=0;}
      if (mCurrentParticleNumberInFile>=mNumberOfParticlesInFile) {mCurrentParticleNumberInFile=0;}
      GenerateROOTVertex( event );
      mCurrentParticleNumberInFile++;
    }
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
    mResidu = mRequestedNumberOfParticlesPerRun-mTotalNumberOfParticles*mLoop;
  }

  mParticleMomentum2 = mParticleMomentum;
  mParticlePosition2 = mParticlePosition;

  if (GetPositionInWorldFrame()) mParticleMomentum2 = SetReferenceMomentum(mParticleMomentum2); //momentum: convert world frame coordinate to local volume coordinate

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

  if (GetPositionInWorldFrame()) mParticlePosition2 = SetReferencePosition(mParticlePosition2); //particle: convert world frame coordinate to local volume coordinate

  if (GetUseRegularSymmetry() && mCurrentUse!=0) { mParticlePosition2  =  rotation*mParticlePosition2; }
  if (GetUseRandomSymmetry() && mCurrentUse!=0) {  mParticlePosition2 =  rotation*mParticlePosition2; }

  ChangeParticlePositionRelativeToAttachedVolume(mParticlePosition2);


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

  if (listOfPhaseSpaceFile.size()==0){
    if (extension == "root") mFileType = "rootFile";
    else if (extension == "IAEAphsp" || extension == "IAEAheader" ) mFileType = "IAEAFile";
    else GateError( "Unknow phase space file extension. Knowns extensions are : "
                    << Gateendl << ".IAEAphsp (or IAEAheader), .root\n");
    listOfPhaseSpaceFile.push_back(file);
    return;
  }

  if (extension == "root" && mFileType == "rootFile") listOfPhaseSpaceFile.push_back(file);
  else if ((extension == "IAEAphsp" || extension == "IAEAheader") && mFileType == "IAEAFile") listOfPhaseSpaceFile.push_back(file);
  else GateError( "Cannot add phase space files with different extension");

}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
G4ThreeVector GateSourcePhaseSpace::SetReferencePosition(G4ThreeVector coordLocal)
{
  for(int j = mListOfRotation.size()-1;j>=0;j--){
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

#endif
