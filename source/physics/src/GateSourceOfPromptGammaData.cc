/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateSourceOfPromptGammaData.hh"
#include "GateMessageManager.hh"
#include "Randomize.hh" // needed for G4UniformRand
#include "G4Gamma.hh"
#include "GateRandomEngine.hh"
#include <random>
#include <iostream>
#include <fstream>

//------------------------------------------------------------------------
GateSourceOfPromptGammaData::GateSourceOfPromptGammaData()
{
  computesum = 0;
  mTofFlag = false;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
GateSourceOfPromptGammaData::~GateSourceOfPromptGammaData()
{
  for(unsigned int i=0; i< mPositionYGen.size(); i++)
    delete mPositionYGen[i];
  for(unsigned int i=0; i< mPositionZGen.size(); i++)
    for(unsigned int j=0; i< mPositionZGen[i].size(); j++)
      delete mPositionZGen[i][j];
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::LoadData(std::string mFilename)
{
  mImage = new GateImageOfHistograms("float");
  mImage->Read(mFilename);
  
  if (mTofFlag) {
    mImageTof = new GateImageOfHistograms("float");
    mImageTof->Read(G4String(removeExtension(mFilename))+"-tof."+G4String(getExtension(mFilename)));
  }
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SetTof(G4bool newflag)
{
  mTofFlag = newflag;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::Initialize()
{
  unsigned int sizeX = mImage->GetResolution().x();
  unsigned int sizeY = mImage->GetResolution().y();
  unsigned int sizeZ = mImage->GetResolution().z();
  unsigned int nbOfBins = mImage->GetNbOfBins();
  unsigned int nbOfBinsTof = mImageTof->GetNbOfBins();
  unsigned long nbOfValues = sizeX*sizeY*sizeZ;

  // Random generators for position: set sizes
  mPositionYGen.resize(sizeX);
  mPositionZGen.resize(sizeX);
  for(unsigned int i=0; i<sizeX; i++) {
    mPositionZGen[i].resize(sizeY);
  }

  // Build the scalar image with total number of counts at each pixel
  mImage->ComputeTotalOfCountsImageDataFloat(mDataCounts); // "mDataCounts" is a 3D volume of the PG yield integrated of the the PG energy 
  computesum = mImage->ComputeSum(); // = global PG yield per proton
  // no need to do it for tof, mDataCounts is just used for biased spatial random sampling
  
  // Initialize random generator for position. Loop over the total
  // count scalar image (mDataCounts)
  mPositionXGen.SetXBias(G4ThreeVector(0., 0., 0.)); // important
  for(unsigned int i=0; i<sizeX; i++) {
    double sumYZ = 0.0;
    mPositionYGen[i] = new G4SPSRandomGenerator;
    mPositionYGen[i]->SetYBias(G4ThreeVector(0., 0., 0.)); // important
    for(unsigned int j=0; j<sizeY; j++) {
      double sumZ = 0.0;
      mPositionZGen[i][j] = new G4SPSRandomGenerator;
      mPositionZGen[i][j]->SetZBias(G4ThreeVector(0., 0., 0.)); // important
      for(unsigned int k=0; k<sizeZ; k++) {
        double val = mDataCounts[mImage->GetIndexFromPixelIndex(i, j, k)];
        sumZ += val;
        // Bias the Z component according to the voxel value
        mPositionZGen[i][j]->SetZBias(G4ThreeVector(k+1 ,val,0.));
      }
      sumYZ += sumZ;
      // Bias the Y component according to integration over Z
      mPositionYGen[i]->SetYBias(G4ThreeVector(j+1, sumZ,0.));
    }
    // Bias the X component according to integration over YZ plane
    mPositionXGen.SetXBias(G4ThreeVector(i+1, sumYZ,0.));
  }

  // Initialize energy.
  mEnergyGen.resize(nbOfValues);
  double energyStep  = (mImage->GetMaxValue()-mImage->GetMinValue())/nbOfBins;
  double energy = 0.0;
  
  long indexImage = 0;
  long indexData = 0;
  float * data = mImage->GetDataFloatPointer();
  long nbNonZero = 0;

  // We only create TH1D for non zero pixel.
  for(unsigned int k=0; k<sizeZ; k++) {
    for(unsigned int j=0; j<sizeY; j++) {
      for(unsigned int i=0; i<sizeX; i++) {
        if (mDataCounts[indexImage] == 0) { // FIXME
          indexData+=nbOfBins;
        }
        else {
          energy = mImage->GetMinValue() + energyStep/2.; //the energy of each bin corresponds to its center 
          mEnergyGen[indexImage] = new TH1D;
          // This is much much faster to use the constructor without
          // param, the SetBins than using the following line with constructor :
          // new TH1D("", "", nbOfBins, mImage->GetMinValue(), mImage->GetMaxValue());
          TH1D * h = mEnergyGen[indexImage];
          h->SetBins(nbOfBins, mImage->GetMinValue(), mImage->GetMaxValue());
	  // G4cout << "GateSourceOfPromptGammaData::Initialize: lowEdge 1 = " << h->GetXaxis()->GetBinLowEdge(1)
	  // 	 << " -- upEdge " << h->GetXaxis()->GetNbins()
	  // 	 << " = " << h->GetXaxis()->GetBinUpEdge(h->GetXaxis()->GetNbins()) << G4endl;
	  // G4cout << "GateSourceOfPromptGammaData::Initialize: binCenter 1 = " << h->GetXaxis()->GetBinCenter(1)
	  // 	 << " -- binCenter " << h->GetXaxis()->GetNbins()
	  // 	 << " = " << h->GetXaxis()->GetBinCenter(h->GetXaxis()->GetNbins()) << G4endl;
	  for(unsigned int l=0; l<nbOfBins; l++) {
            h->Fill(energy, data[indexData]);
            indexData++;
            energy += energyStep;
          }
          nbNonZero++;
        }
        indexImage++;
      }
    }
  }

  // Initialize pgtime.
  if (mTofFlag) {
    mPgtimeGen.resize(nbOfValues);
    double pgtimeStep  = (mImageTof->GetMaxValue()-mImageTof->GetMinValue())/nbOfBinsTof;
    double pgtime = 0.0;
    long indexImageTof = 0;
    long indexDataTof = 0;
    float * dataTof = mImageTof->GetDataFloatPointer();
    long nbNonZeroTof = 0;

    // We only create TH1D for non zero pixel.
    for(unsigned int k=0; k<sizeZ; k++) {
      for(unsigned int j=0; j<sizeY; j++) {
	for(unsigned int i=0; i<sizeX; i++) {
	  if (mDataCounts[indexImageTof] == 0) { // FIXME
	    indexDataTof+=nbOfBinsTof;
	  }
	  else {
	    pgtime = mImageTof->GetMinValue() + pgtimeStep/2.; // The TH1D bin center must be selected (ie half he width is added)
	    mPgtimeGen[indexImageTof] = new TH1D;
	    // This is much much faster to use the constructor without
	    // param, the SetBins than using the following line with constructor :
	    // new TH1D("", "", nbOfBins, mImage->GetMinValue(), mImage->GetMaxValue());
	    TH1D * hTof = mPgtimeGen[indexImageTof];
	    hTof->SetBins(nbOfBinsTof, mImageTof->GetMinValue(), mImageTof->GetMaxValue());
	    for(unsigned int l=0; l<nbOfBinsTof; l++) {
	      hTof->Fill(pgtime, dataTof[indexDataTof]);
	      indexDataTof++;
	      pgtime += pgtimeStep;
	    }
	    nbNonZeroTof++;
	  }
	  indexImageTof++;
	}
      }
    }
  }

  // Initialize direction sampling
  G4SPSRandomGenerator * biasRndm = new G4SPSRandomGenerator;
  mAngleGen.SetBiasRndm(biasRndm);
  mAngleGen.SetPosDistribution(new G4SPSPosDistribution); // needed
  mAngleGen.SetAngDistType("iso");

  // ATTENTION: THIS DELETES THE ON DISK DATA FROM MEMORY. ACCESSING THE DATA IN mImage
  // WILL SEGFAULT. METADATA IS KEPT.
  mImage->Deallocate();
  if (mTofFlag) mImageTof->Deallocate();
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
  else y = mPositionYGen[i]->GenRandY();
  int j = mCurrentIndex_j = floor(y);
  double z;
  if (mImage->GetResolution().z() == 1) z=G4UniformRand();
  else  z = mPositionZGen[i][j]->GenRandZ();
  mCurrentIndex_k = floor(z);

  // Offset according to image origin (and half voxel position)
  x = mImage->GetOrigin().x() + x*mImage->GetVoxelSize().x();
  y = mImage->GetOrigin().y() + y*mImage->GetVoxelSize().y();
  z = mImage->GetOrigin().z() + z*mImage->GetVoxelSize().z();

  // Vector
  position.setX(x);
  position.setY(y);
  position.setZ(z);
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SampleRandomEnergy(double & energy)
{
  // Get energy spectrum in the current pixel
  long index = mImage->GetIndexFromPixelIndex(mCurrentIndex_i, mCurrentIndex_j, mCurrentIndex_k);

  if (mDataCounts[index] != 0) {
    energy = mEnergyGen[index]->GetRandom();
  }
  else energy = 0.0;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SampleRandomPgtime(double & pgtime)
{
  if (mTofFlag) {
    // Get energy spectrum in the current pixel
    long index = mImageTof->GetIndexFromPixelIndex(mCurrentIndex_i, mCurrentIndex_j, mCurrentIndex_k);

    if (mDataCounts[index] != 0) {
      pgtime = mPgtimeGen[index]->GetRandom();
    }
    else pgtime = 0.0;
  }
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGammaData::SampleRandomDirection(G4ParticleMomentum & direction)
{
  direction = mAngleGen.GenerateOne();
}
//------------------------------------------------------------------------
