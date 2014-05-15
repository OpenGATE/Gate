/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateSourceLinacBeam.hh"
#include "G4PrimaryParticle.hh"
#include "GateSourceLinacBeamMessenger.hh"
#include "G4Event.hh"
#include "G4ios.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "GateMiscFunctions.hh"

//-------------------------------------------------------------------------------------------------
GateSourceLinacBeam::GateSourceLinacBeam(G4String name):GateVSource(name) {
  mSourceFromPhaseSpaceFilename = "bidon";
  mReferencePosition = G4ThreeVector(0,0,0);
  mMessenger = new GateSourceLinacBeamMessenger(this);
  mTimeList.push_back(0);
  mRmaxList.push_back(std::numeric_limits<double>::max());
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateSourceLinacBeam::~GateSourceLinacBeam()
{
  delete mMessenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceLinacBeam::SetReferencePosition(G4ThreeVector p) {
  mReferencePosition = p;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceLinacBeam::SetRmaxFilename(G4String f) {
  mRmaxFilename = f;
  ReadTimeDoubleValue(mRmaxFilename, "Rmax", mTimeList, mRmaxList);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceLinacBeam::SetSourceFromPhaseSpaceFilename(G4String f) {
  mSourceFromPhaseSpaceFilename = f;

  // Create Root file
  mPhaseSpaceFile = new TFile(mSourceFromPhaseSpaceFilename);

  // Numbers of bins in histo
  mNumberOfVolume=3;
  //  mNbOfRadiusBins=200;
  mNbOfRadiusBinsForAngle=40;
  mNbOfEnergyBinsForAngle=20;
  mVolumeNames.push_back("cible");
  mVolumeNames.push_back("colli");
  mVolumeNames.push_back("filtre");

  // Allocate histo
  mHistoRadius.resize(mNumberOfVolume);
  mHistoThetaDirection.resize(mNumberOfVolume);
  mHistoPhiDirection.resize(mNumberOfVolume);
  mHistoEnergy.resize(mNumberOfVolume);

  for (int i=0; i<mNumberOfVolume; i++) {
    mHistoPhiDirection[i].resize(mNbOfRadiusBinsForAngle);
    mHistoThetaDirection[i].resize(mNbOfRadiusBinsForAngle);
    
    for (int j=0; j<mNbOfRadiusBinsForAngle; j++) {
      mHistoPhiDirection[i][j].resize(mNbOfEnergyBinsForAngle);
      mHistoThetaDirection[i][j].resize(mNbOfEnergyBinsForAngle);
    }
  }

  // Read histo for each starting volume
  mHistoVolume = (TH1D*)(mPhaseSpaceFile->GetKey("histoVolumeDepart"))->ReadObj();

  // DD(mHistoVolume->GetBinContent(0));
  //   DD(mHistoVolume->GetBinContent(1));
  //   DD(mHistoVolume->GetBinContent(2));
  //   DD(mHistoVolume->GetBinContent(3));
  
  // Read histo radius for each starting volume
  for (int i=0; i<mNumberOfVolume; i++) {
    // DD(mVolumeNames[i]);
    G4String n = "histoPositionR"+mVolumeNames[i];
    // DD(n);
    mHistoRadius[i] = (TH1D*)(mPhaseSpaceFile->GetKey(n))->ReadObj();
    // PrintHistoInfo(mHistoRadius[i]);
    if (i==0) mNbOfRadiusBins = mHistoRadius[i]->GetNbinsX();
    else {
      if (mHistoRadius[i]->GetNbinsX() != mNbOfRadiusBins) {
        GateError("The histo named " << mHistoRadius[0]->GetName()
                  << " has " << mNbOfRadiusBins << " bins, while the histo named "
                  << mHistoRadius[i]->GetName() << " has " << mHistoRadius[i]->GetNbinsX()
                  << ". It should be the same. Abord");
      }
    }
    mHistoEnergy[i].resize(mNbOfRadiusBins);
  }
  // DD(mNbOfRadiusBins);

  // Read histo energy for each radius, each source
  for (int i=0; i<mNumberOfVolume; i++) {
    for (int j=0; j<mNbOfRadiusBins; j++) {
      G4String n = "histoE"+mVolumeNames[i]+DoubletoString(j);
      // DD(n);
      mHistoEnergy[i][j] = (TH1D*)(mPhaseSpaceFile->GetKey(n))->ReadObj() ;
    }
  }
  // PrintHistoInfo(mHistoEnergy[0][0]);

  // Read histo for each angle, each radius, each source
  for (int i=0; i<mNumberOfVolume; i++) {
    for (int j=0; j<mNbOfRadiusBinsForAngle; j++) {
      for (int k=0; k<mNbOfEnergyBinsForAngle; k++) {
        G4String n = "histoPhi"+mVolumeNames[i]+DoubletoString(j)+"_"+DoubletoString(k);
        // DD(n);
        mHistoPhiDirection [i][j][k] = (TH1D*)(mPhaseSpaceFile->GetKey(n))->ReadObj() ;

        n = "histoDeltaTheta"+mVolumeNames[i]+DoubletoString(j)+"_"+DoubletoString(k);
        // DD(n);
        mHistoThetaDirection [i][j][k] = (TH1D*)(mPhaseSpaceFile->GetKey(n))->ReadObj() ;
      }
    }
  }
  // PrintHistoInfo(mHistoPhiDirection[0][0][0]);
  // PrintHistoInfo(mHistoThetaDirection[0][0][0]);

  // Print
  GateMessage("Beam", 1, "Nb of vol [" << mNumberOfVolume << "]" << G4endl);
  GateMessage("Beam", 1, "Radius    [" << mNumberOfVolume << "][" << mNbOfRadiusBins << "]" << G4endl);
  GateMessage("Beam", 1, "Energy    [" << mNumberOfVolume << "][" << mNbOfRadiusBins << "][" << mHistoEnergy[0][0]->GetNbinsX() << "]" << G4endl);
  GateMessage("Beam", 1, "ThetaDir  [" << mNumberOfVolume << "][" << mNbOfRadiusBinsForAngle << "][" << mNbOfEnergyBinsForAngle << "][" << mHistoThetaDirection[0][0][0]->GetNbinsX() << "]" << G4endl);
  GateMessage("Beam", 1, "PhiDir    [" << mNumberOfVolume << "][" << mNbOfRadiusBinsForAngle << "][" << mNbOfEnergyBinsForAngle << "][" << mHistoPhiDirection[0][0][0]->GetNbinsX() << "]" << G4endl);
}
//-------------------------------------------------------------------------------------------------

// //-------------------------------------------------------------------------------------------------
// void GateSourceLinacBeam::Update() {
//   if (mSourceFromPhaseSpaceFilename == "bidon") {
//     GateError("Error you should provide a root file with 'setSourceFromPhaseSpaceFilename'");
//   }
  
//   // Update current activity according to time
//   GateMessage("Acquisition", 0, "TODO ********** Source <" << m_name << "> update ACTIVITY" << G4endl);
//   GateMessage("Acquisition", 0, "TODO ********** Source <" << m_name << "> update RMAX" << G4endl);
//   GateVSource::Update();
// }
// //-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
double GateSourceLinacBeam::GetRmaxFromTime(double time) {
  int i = GetIndexFromTime(time);
  // DD(i);
  return mRmaxList[i];
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
int GateSourceLinacBeam::GetIndexFromTime(double aTime) {
  // Search for current "time"
  int i=0; 
  while ((i < (int)mTimeList.size()) && (aTime >= mTimeList[i])) {
    i++;
  }
  i--;
  if ((i < 0) && (aTime < mTimeList[0])) {
    GateError("The time list for " << GetName() << " begin with " << mTimeList[0]/s
              << " sec, so I cannot find the time" << aTime/s << " sec." << G4endl);
  }
  return i;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceLinacBeam::GeneratePrimaryVertex(G4Event* evt) {
  //DD(GetNumberOfParticles());
  double volume;
  int bin,bin1,bin2;
  int volumeNumber = 0;
  double angle=0.;
  double r=0.;
  double Rmax = 666.*mm;  // dummy for first pass in loop
  double posX = Rmax+1., posY=Rmax+1., posZ=Rmax+1.;
  double posXabs = Rmax+1, posYabs=Rmax+1;

  // ========================================================
  // Particle POSITION 
  // ========================================================

  // Select a random position until it is in the Rmax
  while (posXabs>Rmax || posYabs>Rmax) {
    // Starting volume
    volume = mHistoVolume->GetRandom();
    bin = 0;
    volumeNumber = 0;
    while(volume>mHistoVolume->GetBinLowEdge(volumeNumber)) volumeNumber++;
    volumeNumber -= 2;
    //DD(volumeNumber);
    assert(volumeNumber >= 0);
    assert(volumeNumber < mNumberOfVolume);
    
    // Set Rmax according to StartingVolume  ****** TODO *********

    Rmax = GetRmaxFromTime(GetTime()); // for cible not filtre/colli YET

    //Rmax = 50.;                      // filtre et colli1
    //Rmax = 100.;                     // filtre et colli1
    //if (volumeNumber==0) Rmax=25.;   // cible  

    // Get angle from (flat random)
    angle = CLHEP::RandFlat::shoot(2*TMath::Pi());
      
    // Get distance from center 
    r = mHistoRadius[volumeNumber]->GetRandom();
    posX = r*cos(angle);
    posY = r*sin(angle);
    if (posX>0.) posXabs=posX; else posXabs=-posX;
    if (posY>0.) posYabs=posY; else posYabs=-posY;
  }
  posZ = 0.0;

  // ========================================================
  // Particle ENERGY 
  // ========================================================

  // Get the correct bin according to r (distance from center)
  bin = 0;
  while(r>mHistoRadius[volumeNumber]->GetBinLowEdge(bin)) bin++;
  bin -= 2;

  // Get the energy according to the distance bin
  mEnergy = mHistoEnergy[volumeNumber][bin]->GetRandom();

  // ========================================================
  // Particle DIRECTION 
  // ========================================================

  // Get the theta direction (TODO)
  angle=angle*180./TMath::Pi();
  bin1=(int)mNbOfRadiusBinsForAngle*r/100; 
  bin2=(int)mNbOfEnergyBinsForAngle*mEnergy/8; 
  // DD(bin1);
  // DD(bin2);
  double ThetaDirection = mHistoThetaDirection[volumeNumber][bin1][bin2]->GetRandom()+angle;
  // DD(ThetaDirection);

  //==========================================================================================
  //Selection de Phi
  //double Phi = mHistoPhiDirection[volumeNumber][bin]->GetRandom();
  double Phi = mHistoPhiDirection[volumeNumber][bin1][bin2]->GetRandom();
  // DD(Phi);

  //if (posXabs>9 || posYabs>9) {
  //if (posXabs>12 || posYabs>12) {

  //DS TODO !!!

  if (volumeNumber==0) {Phi+=14.9*r/78.5;}             // G4cout<<"  Phi+=14.9*r/78.5= "<<Phi<<G4endl;}
  if (volumeNumber==1) {Phi+=17.*r/80.;}               // G4cout<<"  Phi+=17.*r/80.= "<<Phi<<G4endl;}
  if (volumeNumber==2) {Phi+=27.5*r/80.;}              // G4cout<<"  Phi+=27.5*r/80.= "<<Phi<<G4endl;}
  //}
  //else {
  // on utilise une équation de droite représentative, qui est beaucoup plus juste (technique point source, pour chacun des 3 éléments)
  //if (volumeNumber==0) {Phi=14.9*r/78.5;}             // G4cout<<"  Phi+=14.9*r/78.5= "<<Phi<<G4endl;}
  //if (volumeNumber==1) {Phi=17.*r/80.;}               // G4cout<<"  Phi+=17.*r/80.= "<<Phi<<G4endl;}
  //if (volumeNumber==2) {Phi=27.5*r/80.;}              // G4cout<<"  Phi+=27.5*r/80.= "<<Phi<<G4endl;}
  // }


  //==========================================================================================
  // conversion des angles degrés->radians
  angle=angle*TMath::Pi()/180;
  ThetaDirection=ThetaDirection*TMath::Pi()/180;
  Phi=Phi*TMath::Pi()/180;

  //==========================================================================================
  // Calcul du vecteur direction x y z
  double z = -cos(Phi);
  double Rxy = sqrt(1-z*z);
  double x = Rxy * cos(ThetaDirection);
  double y = Rxy * sin(ThetaDirection);

  // DD(x);
  //   DD(y);
  //   DD(z);

  //==========================================================================================
  //==========================================================================================
  //CREATION DU PHOTON ET EMISSION
  //==========================================================================================
  //==========================================================================================
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition* particle_definition
    = particleTable->FindParticle(particleName="gamma");
  
  if (particle_definition==0) return;
  // create a new vertex
  G4ThreeVector position = G4ThreeVector(posX*mm,posY*mm,posZ*mm);
  // DD(position);
  //   DD(mReferencePosition);
  position = position+mReferencePosition;//G4ThreeVector(0,0,1000);
  // DD(position);
  ChangeParticlePositionRelativeToAttachedVolume(position);
  // DD(position);

  G4PrimaryVertex* vertex = new G4PrimaryVertex(position,GetParticleTime());
  vertex->SetWeight(1.);

  // create new primaries and set them to the vertex
  G4double mass =  particle_definition->GetPDGMass();
  G4double energy = mEnergy + mass;
  G4ParticleMomentum particle_momentum_direction = G4ThreeVector(x,y,z);
  ChangeParticleMomentumRelativeToAttachedVolume(particle_momentum_direction);

  G4double pmom = std::sqrt(energy*energy-mass*mass);
  G4double px = pmom*particle_momentum_direction.x();
  G4double py = pmom*particle_momentum_direction.y();
  G4double pz = pmom*particle_momentum_direction.z();

  G4PrimaryParticle* particle =
    new G4PrimaryParticle(particle_definition,px,py,pz);
  particle->SetMass( mass );
  particle->SetCharge( GetParticleDefinition()->GetPDGCharge() );
  particle->SetPolarization(GetParticlePolarization().x(),
                            GetParticlePolarization().y(),
                            GetParticlePolarization().z() );
  vertex->SetPrimary( particle );  
  evt->AddPrimaryVertex( vertex );
  //G4cout<<"AddPrimaryvertex()"<<G4endl;
}
//-------------------------------------------------------------------------------------------------

