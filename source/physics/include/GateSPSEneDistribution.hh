/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/* ---------------------------------------------------------------------------- *
 *                                                                              *
 *  Class Description :                                                         *
 *                                                                              *
 *  To generate the energy of a primary vertex according to the defined         *
 *  distribution                                                                *
 *                                                                              *
 *   Revision 1.5 2014/08/1 Yann PERROT and Simon NICOLAS  LPC Clermont-ferrand *
 *   Solution for generating particles from Energy spectra (discrete spectrum,  *
 *   histogram and linear interpolation)                                        *
 *   Creation of two new methods:                                               *
 *     ConstructUserSpectrum() and GenerateFromUserSpectrum()                   *
 * -----------------------------------------------------------------------------*/

#ifndef GateSPSEneDistribution_h
#define GateSPSEneDistribution_h 1

#include <vector>

#include <G4Types.hh>
#include <G4String.hh>
#include <G4ParticleDefinition.hh>
#include <G4SPSEneDistribution.hh>


class GateSPSEneDistribution : public G4SPSEneDistribution
{
public:
  GateSPSEneDistribution();

  void GenerateFluor18();
  void GenerateOxygen15();
  void GenerateCarbon11();
  void GenerateRangeEnergy();

  // Shoot an energy in previously created probability tables
  void GenerateFromUserSpectrum();

  // Create probability tables
  void BuildUserSpectrum(G4String fileName);

  G4double GenerateOne(G4ParticleDefinition*);

  void SetEnergyRange(G4double r) { mEnergyRange = r; }

private:
  G4double  mParticleEnergy;
  G4double  mEnergyRange;

  G4int     mMode;
  G4int     mDimSpectrum;
  G4double  mSumProba;

  std::vector<G4double> mTabProba;
  std::vector<G4double> mTabSumProba;
  std::vector<G4double> mTabEnergy;
};

#endif  // GateSPSEneDistribution_h
