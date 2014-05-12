/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePETScannerSystem.hh"

#include "G4UnitsTable.hh"

#include "GateClockDependentMessenger.hh"
#include "GateDigitizer.hh"
#include "GateCoincidenceSorter.hh"

// Constructor
GatePETScannerSystem::GatePETScannerSystem(const G4String& itsName)
: GateScannerSystem( itsName)
{

  G4cout << " Constructeur GatePETScannerSystem " << G4endl;
  // Integrate a coincidence sorter into the digitizer
  G4double coincidenceWindow = 10.* ns;
  GateDigitizer* digitizer = GateDigitizer::GetInstance();
  GateCoincidenceSorter* coincidenceSorter = new GateCoincidenceSorter(digitizer,"Coincidences",coincidenceWindow);
  digitizer->StoreNewCoincidenceSorter(coincidenceSorter);
}
