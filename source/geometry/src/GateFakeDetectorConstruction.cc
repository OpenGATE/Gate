/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEFAKEDETECTORCONSTRUCTION_CC
#define GATEFAKEDETECTORCONSTRUCTION_CC


/*
 * \file  GateFakeDetectorConstruction.cc
 * \brief Fake DetectorConstruction class for development
 */

#include "GateFakeDetectorConstruction.hh"

#include "G4SystemOfUnits.hh"
#include "G4Box.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//-----------------------------------------------------------------------------
G4VPhysicalVolume* GateFakeDetectorConstruction::Construct() {

  G4String name, symbol;             //a=mass of a mole;
  G4double a, z, density;            //z=mean number of protons;  

  G4int ncomponents, natoms;
  G4double fractionmass;


  //
  // define Elements
  //

  a = 1.01*g/mole;
  G4Element* H  = new G4Element(name="Hydrogen",symbol="H" , z= 1., a);

  a = 14.01*g/mole;
  G4Element* N  = new G4Element(name="Nitrogen",symbol="N" , z= 7., a);

  a = 16.00*g/mole;
  G4Element* O  = new G4Element(name="Oxygen"  ,symbol="O" , z= 8., a);

  density = 1.290*mg/cm3;
  G4Material* Air = new G4Material(name="Air"  , density, ncomponents=2);
  Air->AddElement(N, fractionmass=0.7);
  Air->AddElement(O, fractionmass=0.3);

  density = 1.000*g/cm3;
  G4Material* H2O = new G4Material(name="Water", density, ncomponents=2);
  H2O->AddElement(H, natoms=2);
  H2O->AddElement(O, natoms=1);


  G4Box * world_box = new G4Box("world_box",200*cm, 200*cm,200*cm);
  G4LogicalVolume * world_log = new G4LogicalVolume(world_box,Air,"world_log");
  G4VPhysicalVolume* world = new G4PVPlacement(0,               // no rotation
					      G4ThreeVector(), // translation position
				              world_log,             // its logical volume
					      "world",         // its name
					      0,               // its mother (logical) volume
					      false,           // no boolean operations
					      0);              // 


  G4Box * target_box = new G4Box("target_box",30*cm, 30*cm,30*cm);
  G4LogicalVolume * target_log = new G4LogicalVolume(target_box,H2O,"target_log");
  /*  G4VPhysicalVolume* target = */
  new G4PVPlacement(0,               // no rotation
		    G4ThreeVector(), // translation position
		    target_log,             // its logical volume
		    "target",         // its name
		    world_log,               // its mother (logical) volume
		    false,           // no boolean operations
		    0);              // 

  return world;
}

#endif /* end #define GATEFAKEDETECTORCONSTRUCTION_CC */

//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
