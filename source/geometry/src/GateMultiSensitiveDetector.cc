/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


//-----------------------------------------------------------------------------
/// \class GateMultiSensitiveDetector
//-----------------------------------------------------------------------------

#ifndef GATESDM_CC
#define GATESDM_CC

#include "GateMultiSensitiveDetector.hh"

//-----------------------------------------------------------------------------
GateMultiSensitiveDetector::GateMultiSensitiveDetector(G4String name)
  :G4VSensitiveDetector(name),GateNamedObject(name)
{  
  pSensitiveDetector = 0;
  pMultiFunctionalDetector = 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMultiSensitiveDetector::~GateMultiSensitiveDetector()
{

  //if (pSensitiveDetector != 0) delete pSensitiveDetector; //already delete by G4 
  if (pMultiFunctionalDetector != 0) delete pMultiFunctionalDetector;
  // pMultiFunctionalDetector = 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::Initialize(G4HCofThisEvent* hcte)
{
  GateDebugMessageInc("SD",4,"GateMultiSenstiveDetector -- Initialize() -- begin"<<G4endl);
  if(pSensitiveDetector) pSensitiveDetector->Initialize(hcte);
  if(pMultiFunctionalDetector) pMultiFunctionalDetector->Initialize(hcte);
  GateDebugMessageDec("SD",4,"GateMultiSenstiveDetector-- Initialize() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::EndOfEvent(G4HCofThisEvent* hcte)
{
  if(pSensitiveDetector) pSensitiveDetector->EndOfEvent(hcte);
  if(pMultiFunctionalDetector) pMultiFunctionalDetector->EndOfEvent(hcte);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::clear()
{
  if(pSensitiveDetector) pSensitiveDetector->clear();
  if(pMultiFunctionalDetector) pMultiFunctionalDetector->clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::DrawAll()
{
  if(pSensitiveDetector) pSensitiveDetector->DrawAll();
  if(pMultiFunctionalDetector) pMultiFunctionalDetector->DrawAll();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::PrintAll()
{
  if(pSensitiveDetector) pSensitiveDetector->PrintAll();
  if(pMultiFunctionalDetector) pMultiFunctionalDetector->PrintAll();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4bool GateMultiSensitiveDetector::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  if(pSensitiveDetector) pSensitiveDetector->Hit(aStep);
  if(pMultiFunctionalDetector) pMultiFunctionalDetector->Hit(aStep);
  return true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::SetMultiFunctionalDetector(G4String detectorName)
{
  pMultiFunctionalDetector = new G4MultiFunctionalDetector(detectorName);
  // G4SDManager::GetSDMpointer()->AddNewDetector(pMultiFunctionalDetector);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiSensitiveDetector::SetActor(GateVActor * actor)
{
  //actor->GetVolume()->GetLogicalVolume()->SetSensitiveDetector(pMultiFunctionalDetector);
  if(actor->GetNumberOfFilters()!=0)
    actor->SetFilter(actor->GetFilterManager());
  pMultiFunctionalDetector ->RegisterPrimitive(actor);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATESENSORMANAGER_CC */
