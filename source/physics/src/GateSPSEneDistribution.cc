/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <cmath>
#include <vector>
#include <fstream>

#include <G4Types.hh>
#include <G4String.hh>
#include <G4SystemOfUnits.hh>
#include <Randomize.hh>
#include <G4ParticleDefinition.hh>

#include "GateMessageManager.hh"
#include "GateMiscFunctions.hh"

#include "GateSPSEneDistribution.hh"

using namespace std;


//-----------------------------------------------------------------------------
GateSPSEneDistribution::GateSPSEneDistribution()
  : G4SPSEneDistribution(), mParticleEnergy(),
    mEnergyRange(), mMode(), mDimSpectrum(),
    mSumProba(), mTabProba(), mTabSumProba(), mTabEnergy()
{
    // Contrary to G4's G4SPSEneDistribution, we decided to initialize
    // the default energy to 0.0 not to 1.0
    SetMonoEnergy(0.0 * MeV);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSEneDistribution::GenerateFluor18()
{
  // Fit parameters for the Fluor18 spectra
  G4double a = 10.2088;
  G4double b = -30.4551;
  G4double c = 28.4376;
  G4double d = -7.9828;
  G4double E;
  G4double u;
  G4double energyF18 = 0.;

  do
    {
      // hard wired constants!
      E = G4RandFlat::shoot(0.511, 1.144);  // Emin = 0.511 ; Emax = 1.144
      u = G4RandFlat::shoot(0.5209);  // Nmin = 0 ; Nmax = 0.5209
      energyF18 = E;
    }
  while ( u > a*E*E*E + b*E*E + c*E + d );

  G4double energyFluor = energyF18 - 0.511;
  mParticleEnergy = energyFluor;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSEneDistribution::GenerateOxygen15()
{
  // Fit parameters for the Oxygen15 spectra
  G4double a = 3.43874;
  G4double b = -9.04016;
  G4double c = -7.71579;
  G4double d = 13.3147;
  G4double e = 32.5321;
  G4double f = -18.8379;
  G4double E;
  G4double u;
  G4double energyO15 = 0.;

  do
    {
      E = CLHEP::RandFlat::shoot( 0.511, 2.249 );  // Emin ; Emax
      u = CLHEP::RandFlat::shoot( 0., 15.88 );   // Nmin ; Nmax
      energyO15 = E;
    }
  while ( u > a*E*E*E*E*E + b*E*E*E*E + c*E*E*E + d*E*E + e*E + f );

  G4double energyOxygen = energyO15 - 0.511;
  mParticleEnergy = energyOxygen;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSEneDistribution::GenerateCarbon11()
{
  // Fit parameters for the Carbon11 spectra
  G4double a = 2.36384;
  G4double b = -1.00671;
  G4double c = -7.07171;
  G4double d = -7.84014;
  G4double e = 26.0449;
  G4double f = -10.4374;
  G4double E;
  G4double u;
  G4double energyC11 = 0.;

  do
    {
      E = CLHEP::RandFlat::shoot( 0.511, 1.47 ) ; // Emin ; Emax
      u = CLHEP::RandFlat::shoot( 0., 2.2 ) ;   // Nmin ; Nmax
      energyC11 = E;
    }
  while ( u > a*E*E*E*E*E + b*E*E*E*E + c*E*E*E + d*E*E + e*E + f );

  G4double energyCarbon = energyC11 - 0.511;
  mParticleEnergy = energyCarbon;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4double GateSPSEneDistribution::GenerateOne( G4ParticleDefinition* a )
{
  if (GetEnergyDisType() == "Fluor18")           GenerateFluor18();
  else if (GetEnergyDisType() == "Oxygen15")     GenerateOxygen15();
  else if (GetEnergyDisType() == "Carbon11")     GenerateCarbon11();
  else if (GetEnergyDisType() == "Range")        GenerateRangeEnergy();
  else if (GetEnergyDisType() == "UserSpectrum") GenerateFromUserSpectrum();
  else mParticleEnergy = G4SPSEneDistribution::GenerateOne(a);

  return mParticleEnergy;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSPSEneDistribution::GenerateRangeEnergy()
{
  mEnergyRange = GetEmax()-GetEmin();
  mParticleEnergy = (GetEmin()  + G4UniformRand() * mEnergyRange);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// BuildSpectrum fonction read a file which contain in first line mode of spectrum (Spectrum line (1), Histogramm (2)
// or linear interpolated spectrum (3)) in first and in second start energy for the spectrum. Start energy it used only
// for histogram mode but you want give default value (0 for example), even if this value don't serve for mode (1) and (3)
// After the first line each line contain in first column energy and in second probability for this energy.
void GateSPSEneDistribution::BuildUserSpectrum(G4String fileName)
{
  std::ifstream inputFile (fileName.data());
  G4int nline = 0;
  if(inputFile) {
    skipComment(inputFile);

    G4String line;
    G4double probaRead;
    G4double energyRead;

    inputFile >> mMode;
    G4double emin;
    inputFile >> emin;
    SetEmin(emin);

    G4int cursorPosition = inputFile.tellg();  // tellg() save file cursor position

    while(getline(inputFile, line)) nline++;  // count number of line of inputFile
    mDimSpectrum = nline - 1;

    // create two tables for energy and probability
    mTabEnergy.resize(mDimSpectrum);
    mTabProba.resize(mDimSpectrum);

    nline = 0;

    inputFile.clear();
    inputFile.seekg(cursorPosition, inputFile.beg);  // return to the 2nd line in the file

    while(nline < mDimSpectrum) {
      inputFile >> energyRead;
      inputFile >> probaRead;

      mTabEnergy[nline] = energyRead;
      mTabProba[nline] = probaRead;
      nline++;
    }

    inputFile.close();

    // Construct probability table
    mSumProba = 0;
    nline = 0;

    switch(mMode) {
    case 1:  // probability table to create discrete spectrum
      GateMessage("Beam", 2, "Reading UserSpectrum: type is 1=Discrete." << Gateendl);
      mTabSumProba.resize(mDimSpectrum);
      while(nline < mDimSpectrum) {
        mSumProba = mSumProba + mTabProba[nline];
        mTabSumProba[nline] = mSumProba;
        nline++;
      }
      GateMessage("Beam", 2, "Reading UserSpectrum done. " << mDimSpectrum << " bins." << Gateendl);
      break;
    case 2:  // probability table to create histogram
      GateMessage("Beam", 2, "Reading UserSpectrum: type is 2=Histogram." << Gateendl);
      mTabSumProba.resize(mDimSpectrum);
      mSumProba = mTabProba[0] * (mTabEnergy[0] - GetEmin());
      mTabSumProba[0] = mSumProba;
      for(nline = 1; nline < mDimSpectrum; nline++) {
        mSumProba += (mTabEnergy[nline] - mTabEnergy[nline - 1]) * mTabProba[nline];
        mTabSumProba[nline] = mSumProba;
      }
      GateMessage("Beam", 2, "Reading UserSpectrum done. " << mDimSpectrum << " bins." << Gateendl);
      break;
    case 3:  // probability table to create interpolated spectrum
      GateMessage("Beam", 2, "Reading UserSpectrum: type is 3=Interpolated." << Gateendl);
      mTabSumProba.resize(mDimSpectrum - 1);
      for(nline = 1; nline < mDimSpectrum; nline++) {
        // increase by integration over energy interval
        mSumProba += (mTabEnergy[nline] - mTabEnergy[nline - 1]) * mTabProba[nline - 1] - 0.5* (mTabEnergy[nline] - mTabEnergy[nline - 1]) * (mTabProba[nline - 1] - mTabProba[nline]);
        mTabSumProba[nline - 1] = mSumProba;
      }
      GateMessage("Beam", 2, "Reading UserSpectrum done. " << mDimSpectrum << " bins." << Gateendl);
      break;
    default:
      G4Exception("GateSPSEneDistribution::BuildUserSpectrum", "BuildUserSpectrum", FatalException, "Spectrum mode is not recognized, check your spectrum file. Use 1,2 or 3 (Discrete/Histogram/Interpolated).");
      break;
    }
  } else {
    std::string s = "The User Spectrum file '" + fileName + "' is not found.";
    G4Exception("GateSPSEneDistribution::BuildUserSpectrum", "BuildUserSpectrum", FatalException, s.c_str());
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Inverse transform sampling
void GateSPSEneDistribution::GenerateFromUserSpectrum()
{
  G4double pEnergy = 0;

  // random cumulative probabability in ]0...1[
  G4double U = G4UniformRand();

  // identify corresponding interval of the tabulated cumulative distribution
  G4int i = 0;
  while( U >= (mTabSumProba[i] / mSumProba) ) i++;

  G4double delta;
  G4double a, b;
  G4double alpha, beta;
  G4double norm;
  G4double X;

  switch(mMode) {
  case 1:
    // discrete spectrum
    pEnergy = mTabEnergy[i];
    break;
  case 2:
    // histogram spectrum:
    // sample from uniform sub-distribution of the intervall
    if(i == 0) {
      pEnergy = G4RandFlat::shoot(GetEmin(), mTabEnergy[0]);
    } else {
      pEnergy = G4RandFlat::shoot(mTabEnergy[i - 1], mTabEnergy[i]);
    }
    break;
  case 3:
    // linear interpolated spectrum:
    // sample from linear sub-distribution of the intervall
    //
    // checking "mTabProba[i + 1] == mTabProba[i]"
    delta = fabs((mTabProba[i + 1] - mTabProba[i]) / (mTabProba[i + 1] + mTabProba[i]));
    if (delta < 1e-9) {
      // for constant probability sample directly
      pEnergy = G4RandFlat::shoot(mTabEnergy[i], mTabEnergy[i + 1]);
    } else {
      a = mTabEnergy[i];
      b = mTabEnergy[i + 1];
      // Now we can safely assume, that alpha != 0
      alpha = (mTabProba[i + 1] - mTabProba[i]) / (mTabEnergy[i + 1] - mTabEnergy[i]);
      beta = mTabProba[i] - alpha * mTabEnergy[i];
      norm = 0.5 * alpha * (b * b - a * a) + beta * (b - a);
      // random cumulative probability in ]0...1[
      U = G4UniformRand();
      // inversion transform sampling
      X = (-beta + sqrt((alpha * a + beta) * (alpha * a + beta) + 2 * alpha * norm * U)) / alpha;
      if((X - a) * (X - b) <= 0) {
        pEnergy = X;
      } else {
        pEnergy = (-beta - sqrt((alpha * a + beta) * (alpha * a + beta) + 2 * alpha * norm * U)) / alpha;
      }
    }
    break;
  default:
    pEnergy = 0;
    break;

  }

  mParticleEnergy = pEnergy;
}
//-----------------------------------------------------------------------------
