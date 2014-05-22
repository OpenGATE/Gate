/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateSurface.hh"
#include "GateSurfaceMessenger.hh"
#include "GateXMLDocument.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

// -----------------------------------------------------------------------------
GateSurface::GateSurface(const G4String& itsName, GateVVolume* inserter) :
  GateClockDependent(itsName,true),
  m_inserter1(inserter), 
  m_inserter2(0),
  m_opticalsurface(0)
{
  m_messenger = new GateSurfaceMessenger(this);
}
// -----------------------------------------------------------------------------
  
// -----------------------------------------------------------------------------
GateSurface::~GateSurface()
{
  DeleteSurfaces();
  if (m_opticalsurface) delete m_opticalsurface;
  if (m_messenger) delete m_messenger;
}
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void GateSurface::SetInserter2(GateVVolume* inserter)
{ m_inserter2 = inserter;}
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void GateSurface::SetOpticalSurfaceName(const G4String& name)
{
  // try to read the new optical surface
  G4OpticalSurface* surf = ReadOpticalSurface(name);
  if (surf)
  {
    // if there already exists an optical surface delete it
    if (m_opticalsurface) delete m_opticalsurface;
    // set the name of the new optical surface
    m_opticalsurfacename = name;
    // set the new optical surface
    m_opticalsurface = surf;
    // rebuild the surfaces
    BuildSurfaces();
  }
  else
  { G4cerr << "The optical surface '" << name << "' could not be created!\n";}
}
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void GateSurface::BuildSurfaces()
{
  if (IsEnabled()&&m_opticalsurface&&m_inserter2)
  {
    // first delete the old surfaces
    DeleteSurfaces();
    // iterate through all the physical volumes of iterator1
    for (G4int i=0; i<m_inserter1->GetVolumeNumber(); i++)
    {
      G4VPhysicalVolume* vol1 = m_inserter1->GetPhysicalVolume(i);
      // iterate through all the physical volumes of iterator2
      for (G4int j=0; j<m_inserter2->GetVolumeNumber(); j++)
      {
	G4VPhysicalVolume*      vol2    = m_inserter2->GetPhysicalVolume(j);
	// create a new surface
	G4LogicalBorderSurface* surface = new G4LogicalBorderSurface(GetObjectName(),vol1, vol2, m_opticalsurface);
	// and add it to the list
	m_surfaces.push_back(surface);
      }
    }
  }
}
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
void GateSurface::DeleteSurfaces()
{
  for (std::vector<G4LogicalBorderSurface*>::iterator p = m_surfaces.begin(); p != m_surfaces.end(); p++)
  {
    G4LogicalBorderSurface* surface = *p;
    *p = 0;
    delete surface;
  }
  m_surfaces.clear();
}
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
G4OpticalSurface* GateSurface::ReadOpticalSurface(const G4String& name) const
{
  G4OpticalSurface* surface = 0;
  // open the xml document containing the surface descriptions
  GateXMLDocument* doc = new GateXMLDocument("./Surfaces.xml");
  // if correctly opened
  if (doc->Ok())
  {
    // try to find the surface in the document
    doc->Enter();
    if (doc->Find("surface",name))
    {
      // if found create the optical surface
      surface = new G4OpticalSurface(name);
      // model is always the unified model
      surface->SetModel(unified);
      // set the type
      G4String type = doc->GetProperty("type");
      if (type=="dielectric_dielectric") surface->SetType(dielectric_dielectric);
      else if (type=="dielectric_metal") surface->SetType(dielectric_metal);
      // set the finish
      G4String finish = doc->GetProperty("finish");
      if (finish=="polished") surface->SetFinish(polished);
      else if (finish=="ground") surface->SetFinish(ground);
      else if (finish=="polishedbackpainted") surface->SetFinish(polishedbackpainted);
      else if (finish=="groundbackpainted") surface->SetFinish(groundbackpainted);
      else if (finish=="polishedfrontpainted") surface->SetFinish(polishedfrontpainted);
      else if (finish=="groundfrontpainted") surface->SetFinish(groundfrontpainted);
      // set sigma alpha
      G4String sigmaalpha = doc->GetProperty("sigmaalpha");
      surface->SetSigmaAlpha(G4UIcmdWithADouble::GetNewDoubleValue(sigmaalpha.c_str())*deg);
      // read the materialpropertiestable
      doc->Enter();
      G4MaterialPropertiesTable* table = ReadMaterialPropertiesTable(doc);
      surface->SetMaterialPropertiesTable(table);
      doc->Leave();
    }
  }
  else
  { G4cerr << "Could not open the Surfaces.xml file, while surfaces are created: no optical properties read!\n";}

  return surface;
}
// -----------------------------------------------------------------------------

#endif
