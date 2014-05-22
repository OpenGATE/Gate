/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEFAKEDETECTORCONSTRUCTION_HH
#define GATEFAKEDETECTORCONSTRUCTION_HH


/*
 * \file  GateFakeDetectorConstruction.hh
 * \brief fGate Fake DetectorConstruction class for development
 */

#include "G4VUserDetectorConstruction.hh"

class G4Box;
class G4PVPlacement;
class G4Material;
class G4LogicalVolume;

class GateFakeDetectorConstruction : public G4VUserDetectorConstruction {
  
public: 
  // Constructor
  GateFakeDetectorConstruction():G4VUserDetectorConstruction() {}
  
  // Destructor
  virtual ~GateFakeDetectorConstruction() {}
  
  // Construct
  virtual G4VPhysicalVolume* Construct();

}; // end class
  
#endif /* end #define GATEFAKEDETECTORCONSTRUCTION_HH */

//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
