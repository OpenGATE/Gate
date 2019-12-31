/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositroniumDecayChannel_hh
#define GatePositroniumDecayChannel_hh

#include "globals.hh"
#include "G4GeneralPhaseSpaceDecay.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Theorem author for oPs decay: Daria Kamińska ( Eur. Phys. J. C (2016) 76:445 )
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Implements decay of positronium ( pPs and oPs ). Provides support for polarization.
 **/
class GatePositroniumDecayChannel : public G4GeneralPhaseSpaceDecay
{
 public:
  
  //Describes for which positronium we need decay
  enum PositroniumKind { NotDefined, ParaPositronium, OrthoPositronium };

  GatePositroniumDecayChannel( const G4String& theParentName, G4double theBR);
  virtual ~GatePositroniumDecayChannel();
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
    * Exquantation: 89.14
  **/
  G4double GetOrthoPsM( const G4double w1, const G4double w2, const G4double w3 ) const;
  /** Calculate polarization orthogonal to momentum direction
   **/
  G4ThreeVector GetPolarization( const G4ThreeVector& momentum ) const;
  /** Generate perpendiculator vector ( to calculate orthogonal polarization )
   **/
  G4ThreeVector GetPerpendicularVector(const G4ThreeVector& v) const;

 protected:
  //Decay constants
  const G4String kParaPositroniumName = "pPs";
  const G4String kOrthoPositroniumName = "oPs";
  const G4String kDaughterName = "gamma";
  const G4int kParaPositroniumAnnihilationGammasNumber = 2;
  const G4int kOrthoPositroniumAnnihilationGammasNumber = 3;
  const G4double kPositroniumMass = 2.0 * electron_mass_c2;
  PositroniumKind fPositroniumKind = PositroniumKind::NotDefined;
  ///This is maxiaml number which can be calculated by function GetOrthoPsM() - based on 10^7 iterations
  const G4double kOrthoPsMMax = 7.65928;
  const G4double kElectronMass = electron_mass_c2; //[MeV]
};

#endif
