/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GatePhantomHit.hh"
#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "GateConfiguration.h"

G4Allocator<GatePhantomHit> GatePhantomHitAllocator;

GatePhantomHit::GatePhantomHit()
{;}

GatePhantomHit::~GatePhantomHit()
{;}

void GatePhantomHit::Draw()
{
  G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
  if(pVVisManager)
  {
    G4Circle circle(m_pos);
    circle.SetScreenSize(0.04);
    circle.SetFillStyle(G4Circle::filled);
    G4Colour colour(1.,0.,0.);
    G4VisAttributes attribs(colour);
    circle.SetVisAttributes(attribs);
    pVVisManager->Draw(circle);
  }
}

void GatePhantomHit::Print()
{
  G4cout << "PhantomHit" << G4endl
	 << " PDGEncoding " << m_PDGEncoding << G4endl
	 << " edep        " << m_edep        << G4endl
	 << " stepLength  " << m_stepLength  << G4endl
	 << " time        " << m_time        << G4endl
	 << " pos         " << m_pos         << G4endl
	 << " process     " << m_process     << G4endl
	 << " trackID     " << m_trackID     << G4endl
	 << " parentID    " << m_parentID    << G4endl
	 << " voxelCoord  " << m_voxelCoordinates   << G4endl
	 << G4endl;
}

// v. cuplov - optical photons
//const G4String GatePhantomHit::theOutputAlias = "PhantomHits";
// v. cuplov - optical photons
