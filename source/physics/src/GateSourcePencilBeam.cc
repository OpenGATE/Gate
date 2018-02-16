/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

//=======================================================
//	Created by Loïc Grevillot
//	General definition of the class
// One has to make a difference between beam parameters and particle parameters:
//  a) The beam is made of particles and is parametrized with distribution parameters.
//  b) Each particle sampled has fixed values.
// This class allows for defining beam parameters and sample particle properties
// The Beam parameters are defined as follows:
//  a) XoY is the particle origin plan
//  b) default beam direction 0 0 1
//  c) direction, emmittance distributions: theta, phi, emmittanceTheta, emmittancePhi
//=> divergence parameters in the XoZ and YoZ plans
//  d) energy distribution
//  e)User can then give the beam origin mX0, mY0, mZ0 and its main direction,
//    thanks to the SetX0, SetY0, SetZ0 and SetRotation methods.
//=======================================================

// gate
#include "GateConfiguration.h"
#include "GateSourcePencilBeam.hh"

// g4
#include "G4Proton.hh"
#include "G4Tokenizer.hh"
#include "G4UnitsTable.hh"

// std
#include <string>

GateSourcePencilBeam::GateSourcePencilBeam(G4String name, bool useMessenger):
  GateVSource(name), mGaussian2DYPhi(NULL), mGaussian2DXTheta(NULL), mGaussianEnergy(NULL)
{
  //Particle Type
  mParticleType="proton";
  mWeight=1.;
  //Particle Properties If GenericIon [C12]
  mAtomicNumber=6;
  mAtomicMass=12;
  mIonCharge=6;
  mIonExciteEnergy=0;
  //Energy
  mEnergy=1.*MeV;
  mSigmaEnergy=0.;
  //Position
  mPosition[0]=0;
  mPosition[1]=0;
  mPosition[2]=0;
  mSigmaX=0.*cm; mSigmaY=0.*cm;
  //Direction
  mSigmaTheta=0.*deg; mSigmaPhi=0.*deg;
  // first rotation possibility, used by GateSourceTPSPencilBeam
  mRotation[0]=0;
  mRotation[1]=0;
  mRotation[2]=0;
  //second rotation possibility
  mRotationAxis[0]=0;
  mRotationAxis[1]=0;
  mRotationAxis[2]=0;
  mRotationAngle=0;
  //Correlation Position/Direction
  mEllipseXThetaArea=1.;  mEllipseYPhiArea=1.;
  mConvergenceX = mConvergenceY = false;
  //Gaussian distribution generation for direction
  mUXTheta = HepVector(2); mUYPhi = HepVector(2);
  mSXTheta = HepSymMatrix(2,0); mSYPhi = HepSymMatrix(2,0);
  // ** Note that the indexing starts from [0] or (1)
  //Gaussian distribution generation for energy
  //Others
  mTestFlag=false;
  mIsInitialized=false;
  mCurrentParticleNumber=0;
  if (useMessenger) pMessenger = new GateSourcePencilBeamMessenger(this);
}

//------------------------------------------------------------------------------------------------------
GateSourcePencilBeam::~GateSourcePencilBeam()
{
  delete pMessenger;
  delete mGaussianEnergy;
  delete mGaussian2DXTheta;
  delete mGaussian2DYPhi;
}

//------------------------------------------------------------------------------------------------------
void GateSourcePencilBeam::SetIonParameter(G4String ParticleParameters){
  // 4 possible arguments are Z, A, Charge, Excite Energy
  G4Tokenizer next(ParticleParameters);
  mAtomicNumber = StoI(next());
  mAtomicMass = StoI(next());
  G4String sQ = next();
  if (sQ.isNull())
    {
      mIonCharge = mAtomicNumber;
    }
  else
    {
      mIonCharge = StoI(sQ);
      sQ = next();
      if (sQ.isNull())
        {
	  mIonExciteEnergy = 0.0;
        }
      else
        {
	  mIonExciteEnergy = StoD(sQ) * keV;
        }
    }
}



//------------------------------------------------------------------------------------------------------
void GateSourcePencilBeam::GenerateVertex( G4Event* aEvent )
{
  static G4ParticleDefinition* particle_definition;

  if (!mIsInitialized){
    // get GATE (initialized) random engine
    CLHEP::HepRandomEngine *engine = GateRandomEngine::GetInstance()->GetRandomEngine();

    //---------SOURCE PARAMETERS - CONTROL ----------------
    if (pi*mSigmaX*mSigmaTheta<mEllipseXThetaArea){
      GateMessage("Beam",0,"Please make sure that the energy used belongs to the beam model energy range."<<Gateendl);
      GateMessage("Beam",0,"Energy "<<mEnergy<<"\tX "<<mSigmaX<<"\tTheta "<<mSigmaTheta<<"\tEmittance "<<mEllipseXThetaArea<<"\n"<<Gateendl);
      GateError("!!! ERROR !!! -> Wrong Source Parameters: EmittanceX-Theta is lower than Pi*SigmaX*SigmaTheta! Please correct it."<<Gateendl);
    }
    if (pi*mSigmaY*mSigmaPhi<mEllipseYPhiArea){
      GateMessage("Beam",0,"Please make sure that the energy used belongs to the beam model energy range."<<Gateendl);
      GateMessage("Beam",0,"Energy "<<mEnergy<<"\tY "<<mSigmaY<<"\tPhi "<<mSigmaPhi<<"\tEmittance "<<mEllipseYPhiArea<<"\n"<<Gateendl);
      GateError("!!! ERROR !!! -> Wrong Source Parameters: EmmittanceY-Phi is lower than Pi*SigmaY*SigmaPhi! Please correct it.\n"<<Gateendl);
    }

    //---------INITIALIZATION - START----------------------
    mIsInitialized=true;

    if (mTestFlag){
      GateMessage("Beam",0,"----------------TEST CONFIG---------------------------" << Gateendl);
      GateMessage("Beam",0,"--PENCIL BEAM PARAMETERS--" << Gateendl);
      GateMessage("Beam",0,"*Energy: E0 = "<<mEnergy<<" MeV   SigmaEnergy = "<<mSigmaEnergy<<" MeV" << Gateendl);
      //GateMessage("Beam",0,"*Position: X0 = "<<mX0<<"   Y0 = "<<mY0<<"   Z0 = "<<mZ0<< Gateendl);
      GateMessage("Beam",0,"*Position: X0 = "<<mPosition[0]<<"   Y0 = "<<mPosition[1]<<"   Z0 = "<<mPosition[2]<< Gateendl);
      GateMessage("Beam",0,"*Position: sigmaX = "<<mSigmaX<<" mm   sigmaY = "<<mSigmaY<<" mm" << Gateendl);
      GateMessage("Beam",0,"*Direction: sigmaTheta = "<<mSigmaTheta<<" rad   sigmaY' = "<<mSigmaPhi<<" rad" << Gateendl);
      GateMessage("Beam",0,"*Correlation: XTheta ellipse emittance:  "<<mEllipseXThetaArea<<" mm.rad  YPhi ellipse emittance: "<<mEllipseYPhiArea<<" mm.rad" << Gateendl);
      GateMessage("Beam",0,"*Correlation: XTheta ellipse rotation DirNorm:  "<<(mConvergenceX?"positive":"negative")<<"   YPhi ellipse rotation DirNorm: "<<(mConvergenceY?"positive":"negative")<< Gateendl);
    }

    // for initialization mu=0 everywhere.
    // Position offset & Rotations are performed later in the code
    mUXTheta(1)=0.;		//mu X
    mUXTheta(2)=0.;		//mu theta
    mUYPhi(1)=0.;		//mu Y
    mUYPhi(2)=0.;		//mu phi

    delete mGaussianEnergy;
    mGaussianEnergy = new RandGauss(*engine, mEnergy, mSigmaEnergy);
  	
    // Notations & Calculations based on Transport code - Beam Phase Space Notations - P35
    double alpha, beta, gamma, epsilon;
    //==============================================================
    // X Theta Phase Space Ellipse
    epsilon=mEllipseXThetaArea/pi;
    if (epsilon==0) {
      GateError("Error Elipse area is 0 !!!" << Gateendl);
    }
    beta=mSigmaX*mSigmaX/epsilon;
    gamma=mSigmaTheta*mSigmaTheta/epsilon;
    alpha=sqrt(beta*gamma-1.);

    if (!mConvergenceX) {alpha=-alpha;}

    mSXTheta(1,1)=beta*epsilon;
    mSXTheta(1,2)=-alpha*epsilon;
    mSXTheta(2,1)=mSXTheta(1,2);
    mSXTheta(2,2)=gamma*epsilon;

    if (mTestFlag){
      GateMessage("Beam",0,"--ELIPSE X-THETA PARAMETERS--" << Gateendl);
      GateMessage("Beam",0,"Outputs - beta "<<beta<<"  gamma "<<gamma<<"   alpha" <<alpha<<"   epsilon" <<epsilon<<Gateendl);
      GateMessage("Beam",0,"Outputs - Xmax² "<<mSXTheta(1,1)<<"  Ymax² "<<mSXTheta(2,2)<<Gateendl);
      GateMessage("Beam",0,"Outputs - beta*gamma-1 = "<<beta*gamma-1.<<Gateendl);
      GateMessage("Beam",0,"Outputs - beta*gamma-alpha*alpha = "<<beta*gamma-alpha*alpha<<Gateendl);
    }

    delete mGaussian2DXTheta;
    mGaussian2DXTheta = new RandMultiGauss(*engine,mUXTheta,mSXTheta);

    //==============================================================
    // Y Phi Phase Space Ellipse
    epsilon=mEllipseYPhiArea/pi;
    beta=mSigmaY*mSigmaY/epsilon;
    if (epsilon==0) {
      GateError("Error Elipse area is 0 !!!" << Gateendl);
    }
    gamma=mSigmaPhi*mSigmaPhi/epsilon;
    alpha=sqrt(beta*gamma-1.);

    if (!mConvergenceY) {alpha=-alpha;}

    mSYPhi(1,1)=beta*epsilon;
    mSYPhi(1,2)=-alpha*epsilon;
    mSYPhi(2,1)=mSYPhi(1,2);
    mSYPhi(2,2)=gamma*epsilon;

    if (mTestFlag){
      GateMessage("Beam",0,"--ELIPSE Y-PHI PARAMETERS--\n");
      GateMessage("Beam",0,"Outputs - beta "<<beta<<"  gamma "<<gamma<<"   alpha" <<alpha<<"   epsilon" <<epsilon<<Gateendl);
      GateMessage("Beam",0,"Outputs - Xmax² "<<mSYPhi(1,1)<<"  Ymax² "<<mSYPhi(2,2)<<Gateendl);
      GateMessage("Beam",0,"Outputs - beta*gamma-1 = "<<beta*gamma-1.<<Gateendl);
      GateMessage("Beam",0,"Outputs - beta*gamma-alpha*alpha = "<<beta*gamma-alpha*alpha<<Gateendl);
    }
    delete mGaussian2DYPhi;
    mGaussian2DYPhi = new RandMultiGauss(*engine,mUYPhi,mSYPhi);

    //---------INITIALIZATION - END-----------------------


    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4IonTable* ionTable = G4IonTable::GetIonTable();

    if ( mParticleType == "GenericIon" ){
      particle_definition=  ionTable->GetIon( mAtomicNumber, mAtomicMass, mIonExciteEnergy);
      GateMessage("Beam",3, "mParticleType  "<<mParticleType<<"     selected loop \"GenericIon\"" << Gateendl);
      GateMessage("Beam",3,mAtomicNumber<<"  "<<mAtomicMass<<"  "<<mIonCharge<<"  "<<mIonExciteEnergy<< Gateendl);
    }
    else{
      particle_definition = particleTable->FindParticle(mParticleType);
      GateMessage("Beam",3, "mParticleType  "<<mParticleType<<"     selected loop \"other\"" << Gateendl);
    }

    if(particle_definition==0){
      GateError("ERROR: UNSUPPORTED PARTICLE TYPE: " << mParticleType << Gateendl);
    }
  }

  //=======================================================

  //-------- PARTICLE SAMPLING - START------------------
  G4ThreeVector Pos, Dir;
  double energy;

  //energy sampling
  energy = mGaussianEnergy->fire();

  //position/direction sampling
  HepVector XTheta = mGaussian2DXTheta->fire();
  HepVector YPhi = mGaussian2DYPhi->fire();

  Pos[2]=0;			//Pz
  Pos[0]=XTheta(1);		//Px
  Pos[1]=YPhi(1);		//Py

  Dir[2]=1;			//Dz
  Dir[0]=tan(XTheta(2));	//Dx
  Dir[1]=tan(YPhi(2));		//Dy

  // config test direction
  if (mTestFlag){
    GateMessage("Beam",0,"===================" << Gateendl);
    GateMessage("Beam",0,"--SPOT GENERATION--" << Gateendl);
    GateMessage("Beam",0,"°Initial Position        "<<Pos[0]<<"  "<<Pos[1]<<"  "<<Pos[2]<< Gateendl);
    GateMessage("Beam",0,"°Initial Direction       "<<Dir[0]<<"  "<<Dir[1]<<"  "<<Dir[2]<< Gateendl);
  }

  //Rotation and position are performed so that user defines the beam at 0,0,0, with beam direction +Z.
  //Then the beam is rotated around a given axis to set the desired direction.
  //Finally the beam position is set at the desired coordinates.

  // rotations
  // first rotation possibility, used by GateSourceTPSPencilBeam
  Dir.rotateX(mRotation[0]); Dir.rotateY(mRotation[1]); Dir.rotateZ(mRotation[2]);
  Pos.rotateX(mRotation[0]); Pos.rotateY(mRotation[1]); Pos.rotateZ(mRotation[2]);
  //second rotation possibility, using the messenger
  Dir.rotate(mRotationAngle, mRotationAxis);
  Pos.rotate(mRotationAngle, mRotationAxis);

  if (mTestFlag){
    GateMessage("Beam",0,"-AFTER ROTATION \n");
    GateMessage("Beam",0,"°Intermediate Position   "<<Pos[0]<<"  "<<Pos[1]<<"  "<<Pos[2]<< Gateendl);
    GateMessage("Beam",0,"°Final Direction         "<<Dir[0]<<"  "<<Dir[1]<<"  "<<Dir[2]<< Gateendl);
  }
  // initial position offset
  Pos[0]+=mPosition[0];
  Pos[1]+=mPosition[1];
  Pos[2]+=mPosition[2];


  if (mTestFlag){
    GateMessage("Beam",0,"-AFTER POSITION OFFSET \n");
    GateMessage("Beam",0,"°Final Position   "<<Pos[0]<<"  "<<Pos[1]<<"  "<<Pos[2]<< Gateendl);
    //GateMessage("Beam",0,"°Final Direction  "<<Dir[0]<<"  "<<Dir[1]<<"  "<<Dir[2]<< Gateendl);
  }

  //-------- PARTICLE SAMPLING - END------------------

  //=======================================================

  //-------- PARTICLE GENERATION - START------------------
  G4PrimaryVertex* vertex;
  vertex = new G4PrimaryVertex(Pos, mparticle_time);
  vertex->SetWeight(mWeight);

  double mass =  particle_definition->GetPDGMass();
  double energyTot = energy + mass;

  double dtot = std::sqrt(Dir[0]*Dir[0] + Dir[1]*Dir[1] + Dir[2]*Dir[2]);

  double pmom = std::sqrt(energyTot*energyTot-mass*mass);
  double px = pmom*Dir[0]/dtot ;
  double py = pmom*Dir[1]/dtot ;
  double pz = pmom*Dir[2]/dtot ;

  G4PrimaryParticle* particle =  new G4PrimaryParticle(particle_definition,px,py,pz);
  vertex->SetPrimary( particle );
  aEvent->AddPrimaryVertex( vertex );
  mCurrentParticleNumber++;
  //-------- PARTICLE GENERATION - END------------------
}

//------------------------------------------------------------------------------------------------------
G4int GateSourcePencilBeam::GeneratePrimaries( G4Event* event )
{
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << Gateendl);
  G4int numVertices = 0;
  GenerateVertex( event );

  G4PrimaryParticle  * p = event->GetPrimaryVertex(0)->GetPrimary(0);
  GateMessage("Beam", 5, "(" << event->GetEventID() << ") " << p->GetG4code()->GetParticleName()
              << " pos=" << event->GetPrimaryVertex(0)->GetPosition()
              << " weight=" << p->GetWeight()
              << " energy=" <<  G4BestUnit(mEnergy, "Energy")
              << " mom=" << p->GetMomentum()
              << " ptime=" <<  G4BestUnit(p->GetProperTime(), "Time")
              << " atime=" <<  G4BestUnit(GetTime(), "Time")
              << ")" << Gateendl);

  numVertices++;
  return numVertices;
}
// vim: ai sw=2 ts=2 et
