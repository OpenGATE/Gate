/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePromptGammaSpatialEmissionDistribution.hh"
#include "GateMessageManager.hh"

//------------------------------------------------------------------------
GatePromptGammaSpatialEmissionDistribution::GatePromptGammaSpatialEmissionDistribution()
{
  DD("GPGSED::Constructor");
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
GatePromptGammaSpatialEmissionDistribution::~GatePromptGammaSpatialEmissionDistribution()
{
  DD("GPGSED::Destructor");
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GatePromptGammaSpatialEmissionDistribution::SampleRandomPosition(G4ThreeVector & position)
{
  DD("GPGSED::SampleRandomPosition");
  /*
  // Position (in pixel)
  int i = mCurrentIndex_i = floor(mPositionXGen.GenRandX());
  int j = mCurrentIndex_j = floor(mPositionYGen[i].GenRandY());
  int k = mCurrentIndex_k = floor(mPositionZGen[i][k].GenRandZ());
  std::cout << "ijk = " << i << " " << j << " " << k << std::endl;

  // Position (in physical units ; uniform rand in pixel)
  //  FIXME --> compte coord from index
  double x =

  double x = mIndexCoordX[i] + (G4UniformRand()-0.5)*mVoxelSize.x();
  double y = mIndexCoordY[j] + (G4UniformRand()-0.5)*mVoxelSize.y();
  double z = mIndexCoordZ[k] + (G4UniformRand()-0.5)*mVoxelSize.z();
  std::cout << "xyz = " << x << " " << y << " " << z << std::endl;

  // Vector
  position.setX(x);
  position.setY(y);
  position.setZ(z);

  // Rotation and translation FIXME
  DD("TODO FIXME position Rotation translation");
  // ChangeParticleMomentumRelativeToAttachedVolume ??
  */
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GatePromptGammaSpatialEmissionDistribution::SampleRandomEnergy(double & energy)
{
  DD("GPGSED::SampleRandomEnergy");

  // Get energy in the bins (no interpolation, yet) FIXME
  //  energy = mEnergyGen[i][j][k].GenRandEnergy();
  DD(energy);

}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GatePromptGammaSpatialEmissionDistribution::SampleRandomDirection(G4ParticleMomentum & direction)
{
  DD("GPGSED::SampleRandomDirection");

  //  direction = mAngleGen->GenerateOne();

  // Rotation and translation FIXME
  DD("TODO FIXME direction Rotation translation");

}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GatePromptGammaSpatialEmissionDistribution::Initialize()
{
  DD("GPGSED::Initialize");
  /*
  // Random generators for position: set sizes
  mPositionYGen.resize(mSizeX);
  mPositionZGen.resize(mSizeX);
  for(unsigned int i=0; i<mSizeX; i++) mPositionZGen[i].resize(mSizeY);

  // Build the scalar image with total number of count at each voxel
  //FIXME

  // Loop over scalar image (total counts at each voxel)
  for(unsigned int i=0; i<mSizeX; i++) {
    double sumYZ = 0.0;
    for(unsigned int j=0; j<mSizeY; j++) {
      double sumZ = 0.0;
      for(unsigned int k=0; k<mSizeZ; k++) {
        double val = mDistrib.GetValue(i,j,k);
        sumZ += val;
        // Bias the Z component according to the voxel value
        mPositionZGen[i][j].SetZBias(G4ThreeVector(k+1,val,0.));
      }
      sumYZ += sumZ;
      // Bias the Y component according to integration over Z
      mPositionYGen[i].SetYBias(G4ThreeVector(j+1,sumZ,0.));
    }
    // Bias the X component according to integration over YZ plane
    mPositionXGen.SetXBias(G4ThreeVector(i+1,sumYZ,0.));
  }

  */
}
//------------------------------------------------------------------------

  /*
    mDistrib->SetRandomEngine(engine);
    mDistrib->LoadData(mFilename);
    mDistrib->Initialize();
  */
