/*----------------------
  GATE version name: gate_v...

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATESOURCETPSPENCILBEAM_HH
#define GATESOURCETPSPENCILBEAM_HH

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
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
#include "GateSourceTPSPencilBeamMessenger.hh"
#include "GateSourcePencilBeam.hh"

#include "CLHEP/Random/RandGeneral.h"
#include "CLHEP/RandomObjects/RandMultiGauss.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "GateRandomEngine.hh"
#include "TMath.h"

void ReadLineTo3Doubles(double *toto, char *oneline);

//------------------------------------------------------------------------------------------------------
class GateSourceTPSPencilBeam : public GateVSource
{
public:

  typedef CLHEP::RandGeneral RandGeneral;

  GateSourceTPSPencilBeam( G4String name);
  ~GateSourceTPSPencilBeam();

  G4int GeneratePrimaries( G4Event* event );
  void GenerateVertex( G4Event* );

  //Particle Type
  void SetParticleType(G4String ParticleType) {strcpy(mParticleType, ParticleType);}
  //Test Flag
  void SetTestFlag(bool b) {mTestFlag=b;}
  //Treatment Plan file
  void SetPlan(string plan) {mPlan=plan;}
  //FlatGenerationFlag
  void SetGeneFlatFlag(bool b) {mFlatGenerationFlag=b;}
  //Pencil beam parameters calculation
  double GetEnergy(double energy);
  double GetSigmaEnergy(double energy);
  double GetSigmaX(double energy);
  double GetSigmaY(double energy);
  double GetSigmaTheta(double energy);
  double GetSigmaPhi(double energy);
  double GetEllipseXThetaArea(double energy);
  double GetEllipseYPhiArea(double energy);
  //List of not allowed fields
  void SetNotAllowedField (int fieldID) {mNotAllowedFields.push_back(fieldID);}
  // Select a single Layer
  void SelectLayerID (int layerID) { mSelectedLayerID = layerID;}
  //MU to Protons conversion
  double ConvertMuToProtons(double weight, double energy);
  //select beam descriptionfile
  void SetSourceDescriptionFile(G4String FileName){mSourceDescriptionFile=FileName; mIsASourceDescriptionFile=true;}
  //load plan description file
  void LoadClinicalBeamProperties();
  //Configuration of spot intensity
  void SetSpotIntensity(bool b) {mSpotIntensityAsNbProtons=b;}
  //Convergent or divergent beam model
  void SetBeamConvergence(bool c) {mConvergentSource=c;}

protected:

  GateSourceTPSPencilBeamMessenger * pMessenger;

  bool mIsInitialized;
  int mCurrentSpot, mTotalNumberOfSpots;
  bool mIsASourceDescriptionFile;
  G4String mSourceDescriptionFile;

  vector<GateSourcePencilBeam*> mPencilBeams;
  double mDistanceSMXToIsocenter;
  double mDistanceSMYToIsocenter;
  double mDistanceSourcePatient;
  //Particle Type
  char mParticleType[64];
  //Test flag (for verbosity)
  bool mTestFlag;
  //Treatment Plan file
  G4String mPlan;
  //Others
  double mparticle_time ;
  int mCurrentParticleNumber;
  //Distribution of the spot sources
  bool mFlatGenerationFlag;
  double *mPDF;
  RandGeneral * mDistriGeneral;
  //Not alloweed fields
  vector<int> mNotAllowedFields;
  //clinical beam parameters (polynomial equations)
  vector<double> mEnergy, mEnergySpread, mX, mY, mTheta, mPhi, mXThetaEmittance, mYPhiEmittance;
  //Configuration of spot intensity
  bool mSpotIntensityAsNbProtons;
  //Convergent or divergent beam model
  bool mConvergentSource;
  int mSelectedLayerID;
};
//------------------------------------------------------------------------------------------------------


#endif
#endif
