/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About class: Represents a single positronium species (para-Ps or ortho-Ps), wrapping the Geant4 particle decay table to provide access to lifetime, annihilation gamma count, and decay products.
 **/

#ifndef GatePositronium_hh
#define GatePositronium_hh

#include "G4DecayProducts.hh"
#include "GatePositroniumDecayChannel.hh"

class GatePositronium {
public:
  GatePositronium(const G4String& name, G4double life_time);
  ~GatePositronium() = default;

  GatePositronium(const GatePositronium&) = delete;
  GatePositronium& operator=(const GatePositronium&) = delete;
  GatePositronium(GatePositronium&&) = default;
  GatePositronium& operator=(GatePositronium&&) = default;


  G4double GetLifeTime() const;
  const G4String& GetName() const;
  G4int GetAnnihilationGammasNumber() const;
  G4DecayProducts *GetDecayProducts() const;

private:
  G4String fName;
  G4double fLifeTime = 0.0; //[ns]
  GatePositroniumDecayChannel *pDecayChannel = nullptr; // Todo check who owns pDecayChannel?
};
#endif
