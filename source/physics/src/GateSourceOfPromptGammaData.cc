/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateSourceOfPromptGammaData.hh"
#include "GateMessageManager.hh"
#include "Randomize.hh" // needed for G4UniformRand
#include "G4Gamma.hh"
#include "GateRandomEngine.hh"

// FIXME to remove
#include "metaObject.h"
#include "metaImage.h"


//------------------------------------------------------------------------
GateSourceOfPromptGammaData::GateSourceOfPromptGammaData()
{
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
GateSourceOfPromptGammaData::~GateSourceOfPromptGammaData()
{
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::LoadData(std::string mFilename)
{
  DD("GateSourceOfPromptGammaData::LoadData");
  mImage = new GateImageOfHistograms();
  mImage->Read(mFilename);
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::Initialize()
{
  DD("GateSourceOfPromptGammaData::Initialize");
  unsigned int sizeX = mImage->GetResolution().x();
  unsigned int sizeY = mImage->GetResolution().y();
  unsigned int sizeZ = mImage->GetResolution().z();
  unsigned int nbOfBins = mImage->GetNbOfBins();
  unsigned long nbOfValues = sizeX*sizeY*sizeZ;

  // Random generators for position: set sizes
  mPositionYGen.resize(sizeX);
  mPositionZGen.resize(sizeX);
  for(unsigned int i=0; i<sizeX; i++) {
    mPositionZGen[i].resize(sizeY);
  }

  // Build the scalar image with total number of counts at each pixel
  std::vector<double> temp;
  mImage->ComputeTotalOfCountsImageData(temp);

  // Initialize random generator for position. Loop over the total
  // count scalar image (temp)
  mPositionXGen.SetXBias(G4ThreeVector(0., 0., 0.)); // important
  for(unsigned int i=0; i<sizeX; i++) {
    double sumYZ = 0.0;
    mPositionYGen[i].SetYBias(G4ThreeVector(0., 0., 0.)); // important
    for(unsigned int j=0; j<sizeY; j++) {
      double sumZ = 0.0;
      mPositionZGen[i][j].SetZBias(G4ThreeVector(0., 0., 0.)); // important
      for(unsigned int k=0; k<sizeZ; k++) {
        double val = temp[mImage->GetIndexFromPixelIndex(i, j, k)];
        sumZ += val;
        // Bias the Z component according to the voxel value
        mPositionZGen[i][j].SetZBias(G4ThreeVector(k+1 ,val,0.));
      }
      sumYZ += sumZ;
      // Bias the Y component according to integration over Z
      mPositionYGen[i].SetYBias(G4ThreeVector(j+1, sumZ,0.));
    }
    // Bias the X component according to integration over YZ plane
    mPositionXGen.SetXBias(G4ThreeVector(i+1, sumYZ,0.));
  }

  // Initialize energy.
  mEnergyGen.resize(nbOfValues);
  G4SPSRandomGenerator * biasRndm = new G4SPSRandomGenerator;
  for(unsigned int l=0; l<nbOfValues; l++) {
    mEnergyGen[l].SetEnergyDisType("User");
    mEnergyGen[l].SetBiasRndm(biasRndm); // required
  }
  double energyStep  = (mImage->GetMaxValue()-mImage->GetMinValue())/nbOfBins;
  double energy = 0.0;
  long index_image = 0;
  long index_data = 0;
  double * data = mImage->GetDataDoublePointer();
  for(unsigned int k=0; k<sizeZ; k++) {
    for(unsigned int j=0; j<sizeY; j++) {
      for(unsigned int i=0; i<sizeX; i++) {
        energy = mImage->GetMinValue();
        for(unsigned int l=0; l<nbOfBins; l++) {
          G4ThreeVector h;
          h.setX(energy); // energy
          h.setY(data[index_data]); // probability value
          mEnergyGen[index_image].UserEnergyHisto(h);
          index_data++;
          energy += energyStep;
        }
        index_image++;
      }
    }
  }

  // Initialize direction sampling
  mAngleGen.SetBiasRndm(biasRndm);
  mAngleGen.SetPosDistribution(new G4SPSPosDistribution); // needed
  mAngleGen.SetAngDistType("iso");
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SampleRandomPosition(G4ThreeVector & position)
{
  // Random 3D position (in pixel). If size == 1, then bug in GenRand
  // (infinite loop), so we test and set random value [0:1]
  double x;
  if (mImage->GetResolution().x() == 1) x=G4UniformRand();
  else x = mPositionXGen.GenRandX();
  int i =  mCurrentIndex_i = floor(x);
  double y;
  if (mImage->GetResolution().y() == 1) y=G4UniformRand();
  else y = mPositionYGen[i].GenRandY();
  int j = mCurrentIndex_j = floor(y);
  double z;
  if (mImage->GetResolution().z() == 1) z=G4UniformRand();
  else  z = mPositionZGen[i][j].GenRandZ();
  mCurrentIndex_k = floor(z);

  // Offset according to image origin (and half voxel position)
  x = mImage->GetOrigin().x() + x*mImage->GetVoxelSize().x();
  y = mImage->GetOrigin().y() + y*mImage->GetVoxelSize().y();
  z = mImage->GetOrigin().z() + z*mImage->GetVoxelSize().z();

  // Vector
  position.setX(x);
  position.setY(y);
  position.setZ(z);

  // Rotation and translation FIXME
  // DD("TODO FIXME position Rotation translation");
  // ChangeParticleMomentumRelativeToAttachedVolume ??
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SampleRandomEnergy(double & energy)
{
  // Get energy spectrum in the current pixel
  long index = mImage->GetIndexFromPixelIndex(mCurrentIndex_i, mCurrentIndex_j, mCurrentIndex_k);
  energy = mEnergyGen[index].GenerateOne(G4Gamma::Gamma());
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SampleRandomDirection(G4ParticleMomentum & direction)
{
  direction = mAngleGen.GenerateOne();
}
//------------------------------------------------------------------------
