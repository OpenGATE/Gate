/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateSurfaceList.hh"
#include "GateSurfaceListMessenger.hh"
#include "GateVVolume.hh"
#include "GateTools.hh"

GateSurfaceList::GateSurfaceList(GateVVolume* itsCreator, G4bool acceptNewChildren) :
  GateModuleListManager(itsCreator,itsCreator->GetObjectName()+"/surfaces", "surface",false,acceptNewChildren),
  m_messenger(0)
{ if (acceptNewChildren) m_messenger = new GateSurfaceListMessenger(this);}

GateSurfaceList::~GateSurfaceList()
{ if (m_messenger) delete m_messenger;}

void GateSurfaceList::BuildSurfaces()
{ for (size_t i=0; i<size(); i++) GetSurface(i)->BuildSurfaces();}

void GateSurfaceList::AddSurface(GateSurface* surface)
{ theListOfNamedObject.push_back(surface);}

void GateSurfaceList::DescribeSurfaces(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Nb of surfaces:        " << theListOfNamedObject.size() << "\n";
  for (size_t i=0; i<theListOfNamedObject.size(); i++)
  {
    if (theListOfNamedObject[i]) G4cout << GateTools::Indent(indent+1) << "surface: '" << theListOfNamedObject[i]->GetObjectName() << "'\n";
    else G4cout << GateTools::Indent(indent+1) << "detached surface\n";
  }
}

void GateSurfaceList::ListElements()
{ DescribeSurfaces(0);}

#endif
