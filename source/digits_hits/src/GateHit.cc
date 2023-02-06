/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
#include "GateHit.hh"

#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "G4UnitsTable.hh"
#include "G4ios.hh"




G4Allocator<GateHit> GateHitAllocator;

//---------------------------------------------------------------------
GateHit::GateHit()
: m_edep(0),
  m_stepLength(0),
  m_time(0.),
  m_PDGEncoding(0),
  m_trackID(0),
  m_parentID(0),
  m_systemID(-1),
  m_sourceEnergy(-1),
  m_sourcePDG(0),
  m_nCrystalConv(0)
{;}
//---------------------------------------------------------------------


//---------------------------------------------------------------------
GateHit::~GateHit()
{
	;}
//---------------------------------------------------------------------


//---------------------------------------------------------------------
void GateHit::Draw()
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
//---------------------------------------------------------------------


//---------------------------------------------------------------------
void GateHit::Print()
{
  G4cout << this;
  G4cout << Gateendl;
}
//---------------------------------------------------------------------


//---------------------------------------------------------------------
std::ostream& operator<<(std::ostream& flux, const GateHit& hit)
{
  flux   << "("
	 << "E=" << G4BestUnit(hit.m_edep,"Energy") << ", "
	 << "proc=" << hit.m_process << ", "
	 << "particle= " << ( (hit.m_PDGEncoding == 22) ? "gamma" : ( (hit.m_PDGEncoding == 11) ? "e-" : "?" ) ) << ", "
	 << "track=" << hit.m_trackID  << " (son of " << hit.m_parentID    << ") " << ", "
//	 << "outputID= " << hit.GetOutputVolumeID() << ", "
	 << "localPos= [" << G4BestUnit(hit.m_localPos,"Length")    << "], "
	 << "Pos= ["      << G4BestUnit(hit.m_pos,"Length")         << "], "
	 << "Step=  " << G4BestUnit(hit.m_stepLength,"Length")  << ", "
	 << "Time= " << G4BestUnit(hit.m_time,"Time") << ", "
	 << "ScannerPos= ["      << G4BestUnit(hit.m_scannerPos,"Length")         << "], "
	 << "ScannerRotAngle= " << hit.m_scannerRotAngle/degree << " deg"
	 << ")";

  return flux;
}
//---------------------------------------------------------------------


//---------------------------------------------------------------------
std::ofstream& operator<<(std::ofstream& flux, GateHit* hit)
{
  flux   << " " << std::setw(7) << hit->m_runID
	 << " " << std::setw(7) << hit->m_eventID
	 << " " << std::setw(3) << hit->m_primaryID
	 << " " << std::setw(3) << hit->m_sourceID
	 << " " << std::setw(5) << hit->GetOutputVolumeID()
	 << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(30) << std::setprecision(23) << hit->m_time/s
	 << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << hit->m_edep/MeV
	 << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << hit->m_stepLength/mm
	 << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << hit->m_pos.x()/mm
	 << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << hit->m_pos.y()/mm
	 << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << hit->m_pos.z()/mm
	 << " " << std::setw(7) << hit->m_PDGEncoding
	 << " " << std::setw(5) << hit->m_trackID
	 << " " << std::setw(5) << hit->m_parentID
	 << " " << std::setw(3) << hit->m_photonID
	 << " " << std::setw(4) << hit->m_nPhantomCompton
	 << " " << std::setw(4) << hit->m_nPhantomRayleigh
	 << " " << hit->m_process
	 << " " << hit->m_comptonVolumeName
	 << " " << hit->m_RayleighVolumeName
	 << Gateendl;

  return flux;
}
//---------------------------------------------------------------------
