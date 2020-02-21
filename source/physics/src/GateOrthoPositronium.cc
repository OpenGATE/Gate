/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateOrthoPositronium.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4DecayTable.hh"
#include "GatePositroniumDecayChannel.hh"


GateOrthoPositronium* GateOrthoPositronium::theInstance = 0;

GateOrthoPositronium* GateOrthoPositronium::Definition()
{
 if (theInstance !=0) return theInstance;
 
 const G4String name = "oPs";
 const G4double mass = 2.0 * electron_mass_c2;
 const G4int spin = 2;
 const G4int parity = -1;
 const G4double lifetime = 138.6 * ns;
 const G4double BR = 1.0;

 // search in particle table
 G4ParticleTable* pTable = G4ParticleTable::GetParticleTable();
 G4ParticleDefinition* anInstance = pTable->FindParticle(name);

 if ( anInstance == 0 )
 {
  // create particle

  // Arguments for constructor:
  // name, mass, width, charge,
  // spin, parity, C-conjugation,
  // isospin, isospin3, G-parity,
  // type, lepton number, baryon number,
  // PDG encoding, stable, lifetime, 
  // decay table, shortlived, subType
  anInstance = new G4ParticleDefinition( 
                    name, mass, 0.0, 0,
                    spin, parity, 0,
                    0, 0, 0,
                    "lepton", 2, 0,
                    0, false, lifetime,
                    nullptr, false, "e" );

  //create Decay Table 
  G4DecayTable* table = new G4DecayTable();
  // create a decay channel
  G4VDecayChannel* mode = new GatePositroniumDecayChannel( name, BR );
  table->Insert(mode);
  anInstance->SetDecayTable(table);
 }
 theInstance = dynamic_cast<GateOrthoPositronium*>(anInstance);
 return theInstance;
}

GateOrthoPositronium* GateOrthoPositronium::OrthoPositroniumDefinition() { return Definition(); }

GateOrthoPositronium* GateOrthoPositronium::OrthoPositronium() { return Definition(); }
