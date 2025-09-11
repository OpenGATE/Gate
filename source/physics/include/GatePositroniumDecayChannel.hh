/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositroniumDecayChannel_hh
#define GatePositroniumDecayChannel_hh

//#include "globals.hh"
#include "G4GeneralPhaseSpaceDecay.hh"
#include "G4PhysicalConstants.hh"
//#include "G4SystemOfUnits.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Original author of the oPs decay model: Daria Kamińska et al. ( Eur. Phys. J. C (2016) 76:445 )
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Implements pPs and oPs positronium decays. Provides support for polarization.
 **/
class GatePositroniumDecayChannel : public G4GeneralPhaseSpaceDecay
{
 public:
  
  //Describes for which positronium we need decay
  enum PositroniumKind { NotDefined, ParaPositronium, OrthoPositronium };

  GatePositroniumDecayChannel( const G4String& parentName, G4double BR);
  virtual ~GatePositroniumDecayChannel() = default;
  /** Return gammas from positronium decay
  **/
  virtual G4DecayProducts* DecayIt(G4double) override;

 protected:
  /** Return gammas from para-positronium decay
  **/
  G4DecayProducts* DecayParaPositronium();
  /** Return gammas from ortho-positronium decay
  **/
  G4DecayProducts* DecayOrthoPositronium();
  /** Calculate cross section Mij matrix element
    * Based on "Quantum electrodynamics" V. B. BERESTETSKY.
    * Chapter: 89. Annihilation of positronium
    * Equation: 89.14
  **/
  G4double GetOrthoPsM( const G4double w1, const G4double w2, const G4double w3 ) const;
  /** Calculate polarization orthogonal to momentum direction
   **/
  G4ThreeVector GetPolarization( const G4ThreeVector& momentum ) const;
  /** Generate perpendiculator vector ( to calculate orthogonal polarization )
   **/
  G4ThreeVector GetPerpendicularVector(const G4ThreeVector& v) const;

 protected:
  static inline const G4String kParaPositroniumName = "pPs";
  static inline const G4String kOrthoPositroniumName = "oPs";
  static inline const G4String kDaughterName = "gamma";

  //Decay paramters
  static constexpr G4int kParaPositroniumAnnihilationGammasNumber = 2;
  static constexpr G4int kOrthoPositroniumAnnihilationGammasNumber = 3;

  static constexpr G4double kPositroniumMass = 2.0 * electron_mass_c2;
  ///This is maximal number which can be calculated by function GetOrthoPsM() - determined based on 10^7 iterations
  static constexpr G4double kOrthoPsMMax = 7.65928;
  static constexpr G4double kElectronMass = electron_mass_c2; //[MeV]

  PositroniumKind fPositroniumKind = PositroniumKind::NotDefined;
};

#endif
