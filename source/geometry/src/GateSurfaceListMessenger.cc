/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateSurfaceListMessenger.hh"
#include "GateSurfaceList.hh"
#include "GateVVolume.hh"
#include "GateObjectStore.hh"
#include "GateSystemListManager.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"

GateSurfaceListMessenger::GateSurfaceListMessenger(GateSurfaceList* itsChildList) : GateListMessenger(itsChildList)
{ 
  pInsertCmd->SetCandidates(DumpMap());

  pInsertCmd->AvailableForStates(G4State_Idle,G4State_GeomClosed,G4State_EventProc);

}

GateSurfaceListMessenger::~GateSurfaceListMessenger()
{}

const G4String& GateSurfaceListMessenger::DumpMap()
{ 
  static G4String thelist = "";
  return thelist;
}

void GateSurfaceListMessenger::ListChoices()
{
  GateObjectStore* store = GateObjectStore::GetInstance();
  G4cout << "The available volumes are: \n"; 
  for (GateObjectStore::iterator p = store->begin(); p != store->end(); p++)
  { G4cout << "  " << p->second->GetObjectName() << G4endl;}
}

void GateSurfaceListMessenger::DoInsertion(const G4String& surfaceName)
{
  if (GetNewInsertionBaseName().empty()) SetNewInsertionBaseName(surfaceName);
  // check for nameconflicts, modify name when name conflict
  // the following routinge, defined in GateListMessenger uses CheckNameConflict to 
  // determine if there is a name conflict
  AvoidNameConflicts();
  // look up the VObjectInserter
  GateObjectStore* store = GateObjectStore::GetInstance();
  GateVVolume* inserter = store->FindCreator(surfaceName);
  // if the inserter is found, create the surface
  if (inserter)
  {
    // create the surface
    GateSurface* surface = new GateSurface(GetNewInsertionBaseName(), GetCreator());
    surface->SetInserter2(inserter);
    // add it to the surface list
    GetSurfaceList()->AddSurface(surface);
    // rebuild the geometry (this creates the surfaces)
//    TellGeometryToRebuild();
    // reset the insertion basename
    SetNewInsertionBaseName("");
  }
  // if the inserter is not found, do nothing, give warning
  else
  { G4cout << "The volume name '" << surfaceName << "' was not recognised --> insertion request must be ignored!\n";}
}

G4bool GateSurfaceListMessenger::CheckNameConflict(const G4String& name)
{ 
  // look in the surface list for surfaces with the same name
  // when a surface with the same name exists there is a name conflict
  return (GetSurfaceList()->FindSurface(name) != 0);
}


#endif
