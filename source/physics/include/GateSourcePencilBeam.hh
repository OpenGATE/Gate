/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#ifndef GATESOURCEPENCILBEAM_HH
#define GATESOURCEPENCILBEAM_HH

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
#include "G4UImessenger.hh"
#include <iomanip>
#include <vector>

#include "GateVSource.hh"
#include "GateSourcePencilBeamMessenger.hh"

#include "CLHEP/RandomObjects/RandMultiGauss.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "GateRandomEngine.hh"
#include "TMath.h"


class GateSourcePencilBeam : public GateVSource, G4UImessenger
{
  public:

    GateSourcePencilBeam( G4String name);
    ~GateSourcePencilBeam();

    typedef CLHEP::RandMultiGauss RandMultiGauss;
    typedef CLHEP::RandGauss RandGauss;
    typedef CLHEP::HepVector HepVector;
    typedef CLHEP::HepSymMatrix HepSymMatrix;
    typedef CLHEP::HepJamesRandom HepJamesRandom;

    G4int GeneratePrimaries( G4Event* event );
    void GenerateVertex( G4Event* );

    //Particle Type
    void SetParticleType(G4String ParticleType) {strcpy(mParticleType, ParticleType);}
    void SetWeight(double w) {mWeight=w; }
    double GetWeight() {return mWeight; }
    //Particle Properties If GenericIon
    void SetIonParameter(G4String ParticleParameters);
    //Energy
    void SetEnergy(double energy) {mEnergy=energy;}
    void SetSigmaEnergy(double sigmaE) {mSigmaEnergy=sigmaE;}
    //Position
    void SetPosition(G4ThreeVector p) {mPosition=p;}
    void SetSigmaX(double SigmaX) {mSigmaX=SigmaX;}
    void SetSigmaY(double SigmaY) {mSigmaY=SigmaY;}
    //Direction
    void SetSigmaTheta(double SigmaTheta) {mSigmaTheta=SigmaTheta;}
    void SetSigmaPhi(double SigmaPhi) {mSigmaPhi=SigmaPhi;}
    // first rotation possibility => Necessary for the GateSourceTPSPencilBeam !!!!!!
    void SetRotation(G4ThreeVector rot) {mRotation=rot;}
    //second rotation possibility
    void SetRotationAxis(G4ThreeVector axis) {mRotationAxis=axis;}
    void SetRotationAngle(double angle) {mRotationAngle=angle;}
    //Correlation Position/Direction
    void SetEllipseXThetaArea(double EllipseXThetaArea) {mEllipseXThetaArea=EllipseXThetaArea;}
    void SetEllipseYPhiArea(double EllipseYPhiArea) {mEllipseYPhiArea=EllipseYPhiArea;}
    void SetEllipseXThetaRotationNorm(string rotation) {mEllipseXThetaRotationNorm=rotation;}
    void SetEllipseYPhiRotationNorm(string rotation) {mEllipseYPhiRotationNorm=rotation;}
    void SetTestFlag(bool b) {mTestFlag=b;}


  protected:

    GateSourcePencilBeamMessenger * pMessenger;

    bool mIsInitialized;
    //Particle Type
    char mParticleType[64];
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
    string mEllipseXThetaRotationNorm;
    string mEllipseYPhiRotationNorm;
    //Gaussian distribution generation for direction
    RandMultiGauss * mGaussian2DYPhi;
    RandMultiGauss * mGaussian2DXTheta;
    HepVector mUXTheta, mUYPhi;
    HepSymMatrix mSXTheta, mSYPhi;
    //Gaussian distribution generation for energy
    RandGauss * mGaussianEnergy;
    //Others
    bool mTestFlag;
    double mparticle_time;
    int mCurrentParticleNumber;
};

#endif
#endif
