/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
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

#ifndef GATESOURCEPENCILBEAM_CC
#define GATESOURCEPENCILBEAM_CC

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
#include "GateSourcePencilBeam.hh"
#include "G4Proton.hh"
#include "G4Tokenizer.hh"
#include <iostream>
//#include "TMath.h"

GateSourcePencilBeam::GateSourcePencilBeam(G4String name ):GateVSource( name ), mGaussian2DYPhi(NULL), mGaussian2DXTheta(NULL), mGaussianEnergy(NULL)
{ 
  //Particle Type
  strcpy(mParticleType,"proton");
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
  mEllipseXThetaRotationNorm="negative";  mEllipseYPhiRotationNorm="negative";
  //Gaussian distribution generation for direction
  mUXTheta = HepVector(2); mUYPhi = HepVector(2);
  mSXTheta = HepSymMatrix(2,0); mSYPhi = HepSymMatrix(2,0);
  // ** Note that the indexing starts from [0] or (1)
  //Gaussian distribution generation for energy
  //Others
  mTestFlag=false;
  mIsInitialized=false;
  mCurrentParticleNumber=0;
  pMessenger = new GateSourcePencilBeamMessenger(this);
}

//------------------------------------------------------------------------------------------------------
GateSourcePencilBeam::~GateSourcePencilBeam()
{
  delete pMessenger;
  //FIXME segfault when uncommented
  //if (mGaussian2DXTheta) delete mGaussian2DXTheta;
  //if (mGaussian2DYPhi) delete mGaussian2DYPhi;
  //if (mGaussianEnergy) delete mGaussianEnergy;
  // not initialized in this class
  //  delete mEngineXTheta;
  //  delete mEngineYPhi;
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
  if (!mIsInitialized){
    // get GATE (initialized) random engine
    CLHEP::HepRandomEngine *engine = GateRandomEngine::GetInstance()->GetRandomEngine();

    //---------SOURCE PARAMETERS - CONTROL ----------------
    if (TMath::Pi()*mSigmaX*mSigmaTheta<mEllipseXThetaArea){
      cout<<"\n !!! ERROR !!! -> Wrong Source Parameters: EmmittanceX-Theta is lower than Pi*SigmaX*SigmaTheta! Please correct it."<<endl;
      cout<<"Please make sure that the energy used belongs to the beam model energy range."<<endl;
      cout<<"Energy "<<mEnergy<<"\tX "<<mSigmaX<<"\tTheta "<<mSigmaTheta<<"\tEmittance "<<mEllipseXThetaArea<<"\n"<<endl;
      exit(0);
    }
    if (TMath::Pi()*mSigmaY*mSigmaPhi<mEllipseYPhiArea){
      cout<<"\n !!! ERROR !!! -> Wrong Source Parameters: EmmittanceY-Phi is lower than Pi*SigmaY*SigmaPhi! Please correct it.\n"<<endl;
      cout<<"Please make sure that the energy used belongs to the beam model energy range."<<endl;
      cout<<"Energy "<<mEnergy<<"\tY "<<mSigmaY<<"\tPhi "<<mSigmaPhi<<"\tEmittance "<<mEllipseYPhiArea<<"\n"<<endl;
      exit(0);
    }

    //---------INITIALIZATION - START----------------------
    mIsInitialized=true;

    if (mTestFlag){
      G4cout<<"----------------TEST CONFIG---------------------------"<<G4endl;
      G4cout<<"--PENCIL BEAM PARAMETERS--"<<G4endl;
      G4cout<<"*Energy: E0 = "<<mEnergy<<" MeV   SigmaEnergy = "<<mSigmaEnergy<<" MeV"<<G4endl;
      //G4cout<<"*Position: X0 = "<<mX0<<"   Y0 = "<<mY0<<"   Z0 = "<<mZ0<<G4endl;
      G4cout<<"*Position: X0 = "<<mPosition[0]<<"   Y0 = "<<mPosition[1]<<"   Z0 = "<<mPosition[2]<<G4endl;
      G4cout<<"*Position: sigmaX = "<<mSigmaX<<" mm   sigmaY = "<<mSigmaY<<" mm"<<G4endl;
      G4cout<<"*Direction: sigmaTheta = "<<mSigmaTheta<<" rad   sigmaY' = "<<mSigmaPhi<<" rad"<<G4endl;
      G4cout<<"*Correlation: XTheta ellipse emittance:  "<<mEllipseXThetaArea<<" mm.rad  YPhi ellipse emittance: "<<mEllipseYPhiArea<<" mm.rad"<<G4endl;
      G4cout<<"*Correlation: XTheta ellipse rotation DirNorm:  "<<mEllipseXThetaRotationNorm<<"   YPhi ellipse rotation DirNorm: "<<mEllipseYPhiRotationNorm<<"\n"<<G4endl;
    }

    // for initialization mu=0 everywhere.
    // Position offset & Rotations are performed later in the code
    mUXTheta(1)=0.;		//mu X
    mUXTheta(2)=0.;		//mu theta
    mUYPhi(1)=0.;		//mu Y
    mUYPhi(2)=0.;		//mu phi

    mGaussianEnergy = new RandGauss(engine, mEnergy, mSigmaEnergy);

    // Notations & Calculations based on Transport code - Beam Phase Space Notations - P35
    double alpha, beta, gamma, epsilon;
    //==============================================================
    // X Theta Phase Space Ellipse
    epsilon=mEllipseXThetaArea/(TMath::Pi());
    if (epsilon==0) { G4cout<<"Error Elipse area is 0 !!!"<<G4endl;}
    beta=mSigmaX*mSigmaX/epsilon;
    gamma=mSigmaTheta*mSigmaTheta/epsilon;
    alpha=sqrt(beta*gamma-1.);

    if (mEllipseXThetaRotationNorm=="negative") {alpha=-alpha;}

    mSXTheta(1,1)=beta*epsilon;
    mSXTheta(1,2)=-alpha*epsilon;
    mSXTheta(2,1)=mSXTheta(1,2);
    mSXTheta(2,2)=gamma*epsilon;

    if (mTestFlag){
      G4cout<<"--ELIPSE X-THETA PARAMETERS--"<<G4endl;
      G4cout<<"Outputs - beta "<<beta<<"  gamma "<<gamma<<"   alpha" <<alpha<<"   epsilon" <<epsilon<<endl;
      G4cout<<"Outputs - Xmax² "<<mSXTheta(1,1)<<"  Ymax² "<<mSXTheta(2,2)<<endl;
      G4cout<<"Outputs - beta*gamma-1 = "<<beta*gamma-1.<<endl;
      G4cout<<"Outputs - beta*gamma-alpha*alpha = "<<beta*gamma-alpha*alpha<<"\n"<<endl;
    }

    mGaussian2DXTheta = new RandMultiGauss(engine,mUXTheta,mSXTheta);

    //==============================================================
    // Y Phi Phase Space Ellipse
    epsilon=mEllipseYPhiArea/(TMath::Pi());
    beta=mSigmaY*mSigmaY/epsilon;
    if (epsilon==0) {G4cout<<"Error Elipse area is 0 !!!"<<G4endl;}
    gamma=mSigmaPhi*mSigmaPhi/epsilon;
    alpha=sqrt(beta*gamma-1.);

    if (mEllipseYPhiRotationNorm=="negative") {alpha=-alpha;}

    mSYPhi(1,1)=beta*epsilon;
    mSYPhi(1,2)=-alpha*epsilon;
    mSYPhi(2,1)=mSYPhi(1,2);
    mSYPhi(2,2)=gamma*epsilon;

    if (mTestFlag){
      G4cout<<"--ELIPSE Y-PHI PARAMETERS--"<<G4endl;
      G4cout<<"Outputs - beta "<<beta<<"  gamma "<<gamma<<"   alpha" <<alpha<<"   epsilon" <<epsilon<<endl;
      G4cout<<"Outputs - Xmax² "<<mSYPhi(1,1)<<"  Ymax² "<<mSYPhi(2,2)<<endl;
      G4cout<<"Outputs - beta*gamma-1 = "<<beta*gamma-1.<<endl;
      G4cout<<"Outputs - beta*gamma-alpha*alpha = "<<beta*gamma-alpha*alpha<<"\n"<<endl;
    }
    mGaussian2DYPhi = new RandMultiGauss(engine,mUYPhi,mSYPhi);

    //---------INITIALIZATION - END-----------------------
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
    //Pos[0]=1; Pos[1]=2; Pos[2]=1000;
    //Dir[0]=0; Dir[1]=0; Dir[2]=1;
    G4cout<<" "<<G4endl;
    G4cout<<"--SPOT GENERATION--"<<G4endl;
    G4cout<<"°Initial Position        "<<Pos[0]<<"  "<<Pos[1]<<"  "<<Pos[2]<<G4endl;
    G4cout<<"°Initial Direction       "<<Dir[0]<<"  "<<Dir[1]<<"  "<<Dir[2]<<G4endl;
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
    G4cout<<"-AFTER ROTATION "<<G4endl;
    G4cout<<"°Intermediate Position   "<<Pos[0]<<"  "<<Pos[1]<<"  "<<Pos[2]<<G4endl;
    G4cout<<"°Final Direction         "<<Dir[0]<<"  "<<Dir[1]<<"  "<<Dir[2]<<G4endl;
  }
  // initial position offset
  Pos[0]+=mPosition[0];
  Pos[1]+=mPosition[1];
  Pos[2]+=mPosition[2];


  if (mTestFlag){
    G4cout<<"-AFTER POSITION OFFSET "<<G4endl;
    G4cout<<"°Final Position   "<<Pos[0]<<"  "<<Pos[1]<<"  "<<Pos[2]<<G4endl;
    //G4cout<<"°Final Direction  "<<Dir[0]<<"  "<<Dir[1]<<"  "<<Dir[2]<<"\n\n"<<G4endl;
  }

  //-------- PARTICLE SAMPLING - END------------------

  //=======================================================

  //-------- PARTICLE GENERATION - START------------------
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition* particle_definition;

  string parttype=mParticleType;
  if ( parttype == "GenericIon" ){
    particle_definition=  particleTable->GetIon( mAtomicNumber, mAtomicMass, mIonExciteEnergy);
  //G4cout<<G4endl<<G4endl<<"mParticleType  "<<mParticleType<<"     selected loop  GenericIon"<<G4endl;
  //G4cout<<mAtomicNumber<<"  "<<mAtomicMass<<"  "<<mIonCharge<<"  "<<mIonExciteEnergy<<G4endl;
  }
  else{
    particle_definition = particleTable->FindParticle(mParticleType);
  //G4cout<<G4endl<<G4endl<<"mParticleType  "<<mParticleType<<"     selected loop  other"<<G4endl;
  }

  if(particle_definition==0) return;

  G4PrimaryVertex* vertex;
  //  G4ThreeVector particle_position = G4ThreeVector(Pos[0]*mm, Pos[1]*mm, Pos[2]*mm);
  //  vertex = new G4PrimaryVertex(particle_position,mparticle_time);
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
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << G4endl);
  G4int numVertices = 0;
  GenerateVertex( event );

  numVertices++;
  return numVertices;
}

//------------------------------------------------------------------------------------------------------
/*
   G4ThreeVector GateSourcePencilBeam::SetRotation(G4ThreeVector v, double theta, double phi){
   G4double a,b;
   a=v[0]*cos(theta)-v[2]*sin(theta);
   b=v[0]*sin(theta)+v[2]*cos(theta);
   v[0]=a; v[2]=b;

   a=v[1]*cos(phi)-v[2]*sin(phi);
   b=v[1]*sin(phi)+v[2]*cos(phi);
   v[1]=a; v[2]=b;

   return v;
   }
   */

#endif
#endif
