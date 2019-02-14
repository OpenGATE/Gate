/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
#ifndef GATEPHASESPACESOURCE_HH
#define GATEPHASESPACESOURCE_HH

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TH1D.h"
#include "TMath.h"
#include "TKey.h"
#include "TEntryList.h"

#include "G4Event.hh"
#include "globals.hh"
#include "G4VPrimaryGenerator.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleMomentum.hh"
#include <iomanip>
#include <vector>

#include "GateVSource.hh"
#include "GateSourcePhaseSpaceMessenger.hh"

//#include "GateRunManager.hh"

#include "GateUserActions.hh"

struct iaea_record_type;
struct iaea_header_type;

class GateSourcePhaseSpace : public GateVSource
{
public:
  GateSourcePhaseSpace( G4String name);
  ~GateSourcePhaseSpace();

  void Initialize();
  void GenerateROOTVertex( G4Event* );
  void GenerateIAEAVertex( G4Event* );

  G4int OpenIAEAFile(G4String file);

  G4int GeneratePrimaries( G4Event* event );

  void SetSourceInitialization(bool t){mInitialized=t;}
  bool GetSourceInitialization(){return mInitialized;}

  void AddFile(G4String file);

  void InitializeTransformation();

  G4ThreeVector SetReferencePosition(G4ThreeVector coordLocal);
  G4ThreeVector SetReferenceMomentum(G4ThreeVector coordLocal);

  bool GetPositionInWorldFrame(){return mPositionInWorldFrame;}
  void SetPositionInWorldFrame(bool t){mPositionInWorldFrame = t;}

  void SetUseRegularSymmetry(){if(mUseRandomSymmetry) GateError("You cannot use random and regular symmetry for phase space source"); mUseRegularSymmetry = true;}
  bool GetUseRegularSymmetry(){return mUseRegularSymmetry;}
  void SetUseRandomSymmetry(){if(mUseRegularSymmetry) GateError("You cannot use random and regular symmetry for phase space source"); mUseRandomSymmetry = true;}
  bool GetUseRandomSymmetry(){return mUseRandomSymmetry;}

  void SetParticleType(G4String & name) { mParticleTypeNameGivenByUser = name; }

  void SetUseNbOfParticleAsIntensity(bool b) { mUseNbOfParticleAsIntensity = b; }

  void SetRmax(float r){mRmax = r;}

  void SetStartingParticleId(long id) { mStartingParticleId = id; }

protected:

  TChain *T;
  //TEntryList
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

  float energy;
  float x, y, z;
  float dx, dy, dz;
  float ftime;
  double dtime;
  EDataType time_type;
  float weight;

  //  char volumeName;
  char particleName[64];
  G4String mParticleTypeNameGivenByUser;
  double mParticleTime ;//m_source->GetTime();
  G4double mMomentum;

  bool mAlreadyLoad;

  float mRmax;

  double px ;
  double py ;
  double pz ;

  std::vector<G4String> listOfPhaseSpaceFile;

  bool mPositionInWorldFrame;

  FILE* pIAEAFile;
  iaea_record_type *pIAEARecordType;
  iaea_header_type *pIAEAheader;

  G4ParticleDefinition* pParticleDefinition;
  G4PrimaryParticle* pParticle;
  G4PrimaryVertex* pVertex;
  G4ThreeVector mParticlePosition;
  G4ThreeVector mParticleMomentum;
  G4ThreeVector mParticlePosition2;
  G4ThreeVector mParticleMomentum2;

  std::vector<const G4RotationMatrix *> mListOfRotation;
  std::vector<G4ThreeVector> mListOfTranslation;

  bool mUseRegularSymmetry;
  bool mUseRandomSymmetry;
  double mAngle;

  bool mUseNbOfParticleAsIntensity;

};

#endif
#endif
