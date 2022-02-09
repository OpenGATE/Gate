/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATESOURCEPENCILBEAM_HH
#define GATESOURCEPENCILBEAM_HH

#include "GateConfiguration.h"

#include "G4Event.hh"
#include "globals.hh"
#include "G4VPrimaryGenerator.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleMomentum.hh"
#include "G4UImessenger.hh"
#include <iomanip>
#include <vector>
#include <string>

#include "GateVSource.hh"
#include "GateSourcePencilBeamMessenger.hh"

#include "CLHEP/RandomObjects/RandMultiGauss.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "GateRandomEngine.hh"

class GateSourcePencilBeam : public GateVSource, G4UImessenger
{
public:

  GateSourcePencilBeam(G4String name, bool useMessenger=true);
  ~GateSourcePencilBeam();

  typedef CLHEP::RandMultiGauss RandMultiGauss;
  typedef CLHEP::RandGauss RandGauss;
  typedef CLHEP::HepVector HepVector;
  typedef CLHEP::HepSymMatrix HepSymMatrix;
  typedef CLHEP::HepJamesRandom HepJamesRandom;

  G4int GeneratePrimaries( G4Event* event );
  void GenerateVertex( G4Event* );

  //Particle Type
  void SetParticleType(G4String ParticleType) {mIsInitialized &= (ParticleType==mParticleType); mParticleType = ParticleType;}
  void SetWeight(double w) { mWeight=w; } // no need for re-initialization
  double GetWeight() {return mWeight; }
  //Particle Properties If GenericIon
  void SetIonParameter(G4String ParticleParameters);
  //Energy
  void SetEnergy(double energy) {mIsInitialized &= (mEnergy==energy); mEnergy=energy;}
  void SetSigmaEnergy(double sigmaE) {mIsInitialized &= (mSigmaEnergy==sigmaE); mSigmaEnergy=sigmaE;}
  //Position
  void SetPosition(G4ThreeVector p) {mPosition=p;} // no need for re-initialization
  void SetSigmaX(double SigmaX) {mIsInitialized &= (mSigmaX==SigmaX); mSigmaX=SigmaX;}
  void SetSigmaY(double SigmaY) {mIsInitialized &= (mSigmaY==SigmaY); mSigmaY=SigmaY;}
  //Direction
  void SetSigmaTheta(double SigmaTheta) {mIsInitialized &= (mSigmaTheta==SigmaTheta); mSigmaTheta=SigmaTheta;}
  void SetSigmaPhi(double SigmaPhi) {mIsInitialized &= (mSigmaPhi==SigmaPhi); mSigmaPhi=SigmaPhi;}
  // first rotation possibility => Necessary for the GateSourceTPSPencilBeam !!!!!!
  void SetRotation(G4ThreeVector rot) {mRotation=rot;} // no need for re-initialization
  //second rotation possibility
  void SetRotationAxis(G4ThreeVector axis) {mRotationAxis=axis;} // no need for re-initialization
  void SetRotationAngle(double angle) {mRotationAngle=angle;} // no need for re-initialization
  //Correlation Position/Direction
  void SetEllipseXThetaArea(double EllipseXThetaArea) {mIsInitialized &= (mEllipseXThetaArea==EllipseXThetaArea); mEllipseXThetaArea=EllipseXThetaArea;}
  void SetEllipseYPhiArea(double EllipseYPhiArea) {mIsInitialized &= (mEllipseYPhiArea==EllipseYPhiArea); mEllipseYPhiArea=EllipseYPhiArea;}
  void SetEllipseXThetaRotationNorm(std::string rotation) { SetConvergenceX(rotation=="positive");}
  void SetEllipseYPhiRotationNorm(std::string rotation) { SetConvergenceY(rotation=="positive");}
  void SetConvergenceX(bool b) {mIsInitialized &= (mConvergenceX==b); mConvergenceX=b;}
  void SetConvergenceY(bool b) {mIsInitialized &= (mConvergenceY==b); mConvergenceY=b;}
  void SetTestFlag(bool b) {mTestFlag=b;} // no need for re-initialization

protected:
  GateSourcePencilBeamMessenger * pMessenger;

  bool mIsInitialized;
  //Particle Type
  G4String mParticleType;
  double mWeight;
  //Particle Properties If GenericIon
  G4int    mAtomicNumber;
  G4int    mAtomicMass;
  G4int    mIonCharge;
  G4double mIonExciteEnergy;
  //Energy
  double mEnergy;
  double mSigmaEnergy;
  //Position
  G4ThreeVector mPosition;
  double mSigmaX,mSigmaY;
  //Direction
  double mSigmaTheta, mSigmaPhi;
  // first rotation possibility, necessary for the GateSourceTPSPencilBeam !!!!!  no messenger
  G4ThreeVector mRotation;
  //second rotation possibility, with messenger
  G4ThreeVector mRotationAxis;
  double mRotationAngle;
  //Correlation Position/Direction
  double mEllipseXThetaArea;	//mm*rad
  double mEllipseYPhiArea;	//mm*rad
  bool mConvergenceX; // true corresponds to: X-Theta rotation norm is positive
  bool mConvergenceY; // true corresponds to: Y-Phi rotation norm is positive
  //Gaussian distribution generation for direction
  RandMultiGauss * mGaussian2DYPhi;
  RandMultiGauss * mGaussian2DXTheta;
  HepVector mUXTheta, mUYPhi;
  HepSymMatrix mSXTheta, mSYPhi;
  //Gaussian distribution generation for energy
  RandGauss * mGaussianEnergy;
  //Others
  bool mTestFlag;
  double mparticle_time = 0.0;
  int mCurrentParticleNumber;
};

#endif
