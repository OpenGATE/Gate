/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GATESOURCETPSPENCILBEAM_HH
#define GATESOURCETPSPENCILBEAM_HH

// std
#include <iomanip>
#include <vector>

// geant4
#include "G4String.hh"
#include "G4ThreeVector.hh"

// CLHEP
#include "CLHEP/Random/RandGauss.h"

// GATE
#include "GateConfiguration.h"
#include "GateVSource.hh"

class G4Event;
class GateSourceTPSPencilBeamMessenger;
class GateSourcePencilBeam;

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
  void SetParticleType(G4String ParticleType) {mParticleType=ParticleType;}
  //Particle Properties If GenericIon
  void SetIonParameter(G4String ParticleParameters) {mParticleParameters=ParticleParameters;}
  //Specify how to define the particle type
  void SetIsGenericIon(bool IsGenericIon) {mIsGenericIon=IsGenericIon;}
  //Test Flag
  void SetTestFlag(bool b) {mTestFlag=b;}
  //Flag to switch between sorted and random spot selection
  void SetSortedSpotGenerationFlag(bool b) {mSortedSpotGenerationFlag=b;}
  //Flag to switch between absolute (MeV) and relative (%) energy spread specification
  void SetSigmaEnergyInMeVFlag(bool b) {mSigmaEnergyInMeVFlag=b;}
  //Treatment Plan file (plan description file)
  void SetPlan(G4String plan) {mPlan=plan;}
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
  double GetMonitorCalibration(double energy);

  //List of not allowed fields
  void SetNotAllowedField (int fieldID) {mNotAllowedFields.push_back(fieldID);}
  //List of allowed fields
  void SetAllowedField (int fieldID) {mAllowedFields.push_back(fieldID);}
  // Select a single Layer
  void SelectLayerID (int layerID) { mSelectedLayerID = layerID;}
  // Select a single spot
  void SelectSpot (int spot) { mSelectedSpot = spot;}
  //select beam descriptionfile
  void SetSourceDescriptionFile(G4String FileName){mSourceDescriptionFile=FileName; mIsASourceDescriptionFile=true;}
  //load plan description file
  void LoadClinicalBeamProperties();
  //Configuration of spot intensity
  void SetSpotIntensity(bool b) {mSpotIntensityAsNbIons=b;}
  //Convergent or divergent beam model
  void SetBeamConvergence(bool c) {mConvergentSourceXTheta=c; mConvergentSourceYPhi=c;}
  void SetBeamConvergenceXTheta(bool c) {mConvergentSourceXTheta=c;}
  void SetBeamConvergenceYPhi(bool c) {mConvergentSourceYPhi=c;}		
  //Others
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
  double mTotalNbIons;
  bool mIsASourceDescriptionFile;
  G4String mSourceDescriptionFile;

  GateSourcePencilBeam* mPencilBeam;
  double mDistanceSMXToIsocenter;
  double mDistanceSMYToIsocenter;
  double mDistanceSourcePatient;

  //Particle Type
  G4String mParticleType;
  //Particle Properties If GenericIon
  G4String mParticleParameters;
  //ParticleDefinitionMethod;
  bool mIsGenericIon;
  //Test flag (for verbosity)
  bool mTestFlag;

  //generate ions sorted by spot, or randomly
  bool mSortedSpotGenerationFlag;
  //sigma energy definition (MeV or %)
  bool mSigmaEnergyInMeVFlag;
  //Treatment Plan file
  G4String mPlan;
  //Others
  double mparticle_time ;
  int mCurrentParticleNumber;
  //Flag for flat spot delivery
  bool mFlatGenerationFlag;
  //Distribution of the spot sources
  double *mPDF;
  RandGeneral * mDistriGeneral;
  //Disallowed fields
  std::vector<int> mNotAllowedFields;
  //Allowed fields
  std::vector<int> mAllowedFields;
  //clinical beam parameters (polynomial equations)
  std::vector<double> mEnergy, mEnergySpread, mX, mY, mTheta, mPhi, mXThetaEmittance, mYPhiEmittance, mMonitorCalibration;
  //Configuration of spot intensity
  bool mSpotIntensityAsNbIons;
  //Convergent or divergent beam model
  bool mConvergentSourceXTheta, mConvergentSourceYPhi;
  int mSelectedLayerID;
  int mSelectedSpot;
  std::vector<int> mSpotLayer; //in which layer is this spot?
  std::vector<double> mSpotEnergy;
  std::vector<double> mSpotWeight; // (proportional to) the expected number (for each bin in a multinomial distribution)
  std::vector<int> mNbIonsToGenerate; // the actual number (for each bin in a multinomial distribution)
  std::vector<G4ThreeVector> mSpotPosition, mSpotRotation;
};
//------------------------------------------------------------------------------------------------------
// vim: ai sw=2 ts=2 et
#endif
