/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceDigi.hh"

#include "G4UnitsTable.hh"

#include "GatePulse.hh"
#include "GateCoincidencePulse.hh"

#include <iomanip>
#include <fstream>


std::vector<G4bool> GateCoincidenceDigi::m_coincidenceASCIIMask;
G4bool                GateCoincidenceDigi::m_coincidenceASCIIMaskDefault;

G4Allocator<GateCoincidenceDigi> GateCoincidenceDigiAllocator;



GateCoincidenceDigi::GateCoincidenceDigi()
{
  pulseVector[0] = GatePulse();
  pulseVector[1] = GatePulse();
}



GateCoincidenceDigi::GateCoincidenceDigi(GateCoincidencePulse* coincidencePulse)
{
  pulseVector[0] = (*coincidencePulse)[0];
  pulseVector[1] = (*coincidencePulse)[1];
}


GateCoincidenceDigi::GateCoincidenceDigi(const GateCoincidencePulse& coincidencePulse)
{
  pulseVector[0] = coincidencePulse[0];
  pulseVector[1] = coincidencePulse[1];
}


void GateCoincidenceDigi::Draw()
{;}





void GateCoincidenceDigi::Print()
{

  G4cout << this << G4endl;

}



std::ostream& operator<<(std::ostream& flux, GateCoincidenceDigi& digi)
{
  flux    << "GateCoincidenceDigi("
	  << digi.pulseVector[0] << G4endl
	  << digi.pulseVector[1] << G4endl
      	  << ")" << G4endl;

  return flux;
}

std::ofstream& operator<<(std::ofstream& flux, GateCoincidenceDigi* digi)
{
  GatePulse pulse;
  for (G4int iP=0; iP<2; iP++) {
    pulse = digi->GetPulse(iP);
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(0) ) flux << " " << std::setw(7) << pulse.GetRunID();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(1) ) flux << " " << std::setw(7) << pulse.GetEventID();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(2) ) flux << " " << std::setw(5) << pulse.GetSourceID();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(3) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetSourcePosition().x()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(4) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetSourcePosition().y()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(5) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetSourcePosition().z()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(6) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(23) << pulse.GetTime()/s;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(7) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetEnergy()/MeV;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(8) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetGlobalPos().x()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(9) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetGlobalPos().y()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(10) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetGlobalPos().z()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(11) ) flux << " " << std::setw(5) << pulse.GetOutputVolumeID();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(12) ) flux << " " << std::setw(5) << pulse.GetNPhantomCompton();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(13) ) flux << " " << std::setw(5) << pulse.GetNCrystalCompton();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(14) ) flux << " " << std::setw(5) << pulse.GetNPhantomRayleigh();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(15) ) flux << " " << std::setw(5) << pulse.GetNCrystalRayleigh();
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(16) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetScannerPos().z()/mm;
    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(17) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << pulse.GetScannerRotAngle()/deg;;
  }
  flux << G4endl;

  return flux;
}



void GateCoincidenceDigi::SetCoincidenceASCIIMask(G4bool newValue)
{
  m_coincidenceASCIIMaskDefault = newValue;
  for (G4int iMask=0; ((unsigned int) iMask)<m_coincidenceASCIIMask.size(); iMask++) {
    m_coincidenceASCIIMask[iMask] = newValue;
  }
}


void GateCoincidenceDigi::SetCoincidenceASCIIMask(std::vector<G4bool> newMask)
{
  m_coincidenceASCIIMask = newMask;
}

G4bool GateCoincidenceDigi::GetCoincidenceASCIIMask(G4int index)
{
  G4bool mask = m_coincidenceASCIIMaskDefault;
  if ((index >=0 ) && (((unsigned int) index) < m_coincidenceASCIIMask.size())) mask = m_coincidenceASCIIMask[index];
  return mask;
}

std::vector<G4bool> GateCoincidenceDigi::GetCoincidenceASCIIMask()
{
  return m_coincidenceASCIIMask;
}
