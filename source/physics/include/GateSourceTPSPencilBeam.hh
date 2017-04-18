/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATESOURCETPSPENCILBEAM_HH
#define GATESOURCETPSPENCILBEAM_HH

#include "GateConfiguration.h"

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
#include <string>

#include "GateVSource.hh"
#include "GateSourceTPSPencilBeamMessenger.hh"
#include "GateSourcePencilBeam.hh"

#include "CLHEP/Random/RandGeneral.h"
#include "CLHEP/RandomObjects/RandMultiGauss.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "GateRandomEngine.hh"

void ReadLineTo3Doubles(double *toto, const std::string &oneline);

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
  //Temporary Flag to switch between old and new style vertex generation
  void SetOldStyleFlag(bool b) {mOldStyleFlag=b;}
  //Treatment Plan file
  void SetPlan(std::string plan) {mPlan=plan;}
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
  void SetAllowedField (int fieldID) {mAllowedFields.push_back(fieldID);}
  //MU to Protons conversion
  void SelectLayerID (int layerID) { mSelectedLayerID = layerID;}
  //MU to Protons conversion
  void SelectSpot (int spot) { mSelectedSpot = spot;}
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
  int GetCurrentSpotID() {return mCurrentSpot;}
  int GetTotalNumberOfSpots() {return mTotalNumberOfSpots;}
  int GetCurrentLayerID() {return mCurrentLayer;}
  int GetTotalNumberOfLayers() {return mTotalNumberOfLayers;}

protected:

  void ConfigurePencilBeam();

  GateSourceTPSPencilBeamMessenger * pMessenger;

  bool mIsInitialized;
  int mCurrentSpot, mTotalNumberOfSpots;
  int mCurrentLayer, mTotalNumberOfLayers;
  double mTotalNbProtons;
  bool mIsASourceDescriptionFile;
  G4String mSourceDescriptionFile;

  std::vector<GateSourcePencilBeam*> mPencilBeams;
  GateSourcePencilBeam* mPencilBeam; // new style vertex generation uses only one pencil beam
  double mDistanceSMXToIsocenter;
  double mDistanceSMYToIsocenter;
  double mDistanceSourcePatient;
  //Particle Type
  char mParticleType[64];
  //Test flag (for verbosity)
  bool mTestFlag;
  //Old style flag (temporary, for debugging new pencil beam vertex generation)
  bool mOldStyleFlag;
  //Treatment Plan file
  G4String mPlan;
  //Others
  double mparticle_time ;
  int mCurrentParticleNumber;
  //Distribution of the spot sources
  bool mFlatGenerationFlag;
  double *mPDF;
  RandGeneral * mDistriGeneral;
  //Disallowed fields
  std::vector<int> mNotAllowedFields;
  //Allowed fields
  std::vector<int> mAllowedFields;
  //clinical beam parameters (polynomial equations)
  std::vector<double> mEnergy, mEnergySpread, mX, mY, mTheta, mPhi, mXThetaEmittance, mYPhiEmittance;
  //Configuration of spot intensity
  bool mSpotIntensityAsNbProtons;
  //Convergent or divergent beam model
  bool mConvergentSource;
  int mSelectedLayerID;
  int mSelectedSpot;
  std::vector<int> mSpotLayer; //in which layer is this spot?
  std::vector<double> mSpotEnergy;
  std::vector<double> mSpotWeight; // (proportional to) the expected number (for each bin in a multinomial distribution)
  std::vector<int> mNbProtonsToGenerate; // the actual number (for each bin in a multinomial distribution)
  std::vector<G4ThreeVector> mSpotPosition, mSpotRotation;
private:
  void OldGenerateVertex( G4Event* );
  void NewGenerateVertex( G4Event* );
};
//------------------------------------------------------------------------------------------------------

// vim: ai sw=2 ts=2 et
#endif
