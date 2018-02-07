/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*
  \brief Class G4Hybridino : 
  \brief 
*/

#ifndef G4HYBRIDINO_CC
#define G4HYBRIDINO_CC

#include "G4Hybridino.hh"
#include "G4ParticleTable.hh"

//------------------------------------------------------------------------

G4Hybridino* G4Hybridino::theInstance = 0;

//------------------------------------------------------------------------

G4Hybridino* G4Hybridino::Definition()
{
	if (theInstance !=0) return theInstance;
	const G4String name = "hybridino";

	// search in particle table]
	G4ParticleTable* pTable = G4ParticleTable::GetParticleTable();
	G4ParticleDefinition* anInstance = pTable->FindParticle(name);
	if (anInstance ==0)
	{
	// create particle
	//
	//    Arguments for constructor are as follows
	//               name             mass          width         charge
	//             2*spin           parity  C-conjugation
	//          2*Isospin       2*Isospin3       G-parity
	//               type    lepton number  baryon number   PDG encoding
	//             stable         lifetime    decay table
	//             shortlived      subType    anti_encoding
	
	// use constants in CLHEP
	
	anInstance = new G4ParticleDefinition(
      name,            0.0,           0.0,         0,
			0,                 0,             0,          
			0,                 0,             0,             
	      "hybridino",                 0,             0,          0,
		     true,              -1.0,          NULL,
		    false,	  "hybridino"
		);
	
	}

	theInstance = reinterpret_cast<G4Hybridino *>(anInstance);
	return theInstance;
}

//------------------------------------------------------------------------

G4Hybridino*  G4Hybridino::HybridinoDefinition()
{
	return Definition();
}

//------------------------------------------------------------------------

G4Hybridino*  G4Hybridino::Hybridino()
{
	return Definition();
}

//------------------------------------------------------------------------

#endif
