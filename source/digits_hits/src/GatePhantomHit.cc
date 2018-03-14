/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GatePhantomHit.hh"
#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "GateConfiguration.h"
#include "GateMessageManager.hh"

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
  G4cout << "PhantomHit\n"
	 << " PDGEncoding " << m_PDGEncoding << Gateendl
	 << " edep        " << m_edep        << Gateendl
	 << " stepLength  " << m_stepLength  << Gateendl
	 << " time        " << m_time        << Gateendl
	 << " pos         " << m_pos         << Gateendl
	 << " process     " << m_process     << Gateendl
	 << " trackID     " << m_trackID     << Gateendl
	 << " parentID    " << m_parentID    << Gateendl
	 << " voxelCoord  " << m_voxelCoordinates   << Gateendl
	 << Gateendl;
}

// v. cuplov - optical photons
//const G4String GatePhantomHit::theOutputAlias = "PhantomHits";
// v. cuplov - optical photons
