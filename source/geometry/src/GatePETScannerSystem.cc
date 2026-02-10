/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GatePETScannerSystem.hh"

#include "G4UnitsTable.hh"

#include "GateClockDependentMessenger.hh"
#include "GateDigitizerMgr.hh"
#include "GateCoincidenceSorter.hh"

// Constructor
GatePETScannerSystem::GatePETScannerSystem(const G4String& itsName)
: GateScannerSystem( itsName)
{

  //G4cout << " Constructeur GatePETScannerSystem \n";
  // Integrate a coincidence sorter into the digitizer
  //OK GND 2022
  // MR 2025: For multi-detector geometries: suppress the generation of multiple coincidence sorters called
  // "Coincidences" to avoid ambiguity issues
  GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
  if (digitizerMgr->m_CoincidenceSortersList.size() == 0) {
    GateCoincidenceSorter* coincidenceSorter = new GateCoincidenceSorter(digitizerMgr,"Coincidences");
    digitizerMgr->AddNewCoincidenceSorter(coincidenceSorter);
  }
}
