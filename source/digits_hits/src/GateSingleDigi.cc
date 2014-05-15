/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateSingleDigi.hh"
#include "G4UnitsTable.hh"
#include "GatePulse.hh"
#include <iomanip>

std::vector<G4bool> GateSingleDigi::m_singleASCIIMask;
G4bool                GateSingleDigi::m_singleASCIIMaskDefault;

G4Allocator<GateSingleDigi> GateSingleDigiAllocator;




GateSingleDigi::GateSingleDigi()
  : m_pulse()
{
}



GateSingleDigi::GateSingleDigi(GatePulse* pulse)
  : m_pulse(*pulse)
{
}


GateSingleDigi::GateSingleDigi(const GatePulse& pulse)
  : m_pulse(pulse)
{
}




void GateSingleDigi::Draw()
{;}





void GateSingleDigi::Print()
{

  G4cout << this << G4endl;

}



std::ostream& operator<<(std::ostream& flux, const GateSingleDigi& digi)
{
  flux    << "GateSingleDigi(" << G4endl
	  << digi.m_pulse
      	  << ")" << G4endl;

  return flux;
}

std::ofstream& operator<<(std::ofstream& flux, GateSingleDigi* digi)
{
  if ( GateSingleDigi::GetSingleASCIIMask(0) ) flux << " " << std::setw(7) << digi->GetRunID();
  if ( GateSingleDigi::GetSingleASCIIMask(1) ) flux << " " << std::setw(7) << digi->GetEventID();
  if ( GateSingleDigi::GetSingleASCIIMask(2) ) flux << " " << std::setw(5) << digi->GetSourceID();
  if ( GateSingleDigi::GetSingleASCIIMask(3) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3) << digi->GetSourcePosition().x()/mm;
  if ( GateSingleDigi::GetSingleASCIIMask(4) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3) << digi->GetSourcePosition().y()/mm;
  if ( GateSingleDigi::GetSingleASCIIMask(5) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3) << digi->GetSourcePosition().z()/mm;
  if ( GateSingleDigi::GetSingleASCIIMask(6) ) flux << " " << std::setw(5) << digi->GetOutputVolumeID();
  if ( GateSingleDigi::GetSingleASCIIMask(7) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(30) << std::setprecision(23) << digi->GetTime()/s;
  if ( GateSingleDigi::GetSingleASCIIMask(8) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetEnergy()/MeV;
  if ( GateSingleDigi::GetSingleASCIIMask(9) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetGlobalPos().x()/mm;
  if ( GateSingleDigi::GetSingleASCIIMask(10) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetGlobalPos().y()/mm;
  if ( GateSingleDigi::GetSingleASCIIMask(11) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetGlobalPos().z()/mm;
  if ( GateSingleDigi::GetSingleASCIIMask(12) ) flux << " " << std::setw(4) << digi->GetNPhantomCompton();
  if ( GateSingleDigi::GetSingleASCIIMask(13) ) flux << " " << std::setw(4) << digi->GetNCrystalCompton();
  if ( GateSingleDigi::GetSingleASCIIMask(14) ) flux << " " << std::setw(4) << digi->GetNPhantomRayleigh();
  if ( GateSingleDigi::GetSingleASCIIMask(15) ) flux << " " << std::setw(4) << digi->GetNCrystalRayleigh();
  if ( GateSingleDigi::GetSingleASCIIMask(16) ) flux << " " << digi->GetComptonVolumeName();
  if ( GateSingleDigi::GetSingleASCIIMask(17) ) flux << " " << digi->GetRayleighVolumeName();
  flux << G4endl;

  return flux;
}



void GateSingleDigi::SetSingleASCIIMask(G4bool newValue)
{
  m_singleASCIIMaskDefault = newValue;
  for (G4int iMask=0; ((unsigned int)iMask)<m_singleASCIIMask.size(); iMask++) {
    m_singleASCIIMask[iMask] = newValue;
  }
}


void GateSingleDigi::SetSingleASCIIMask(std::vector<G4bool> newMask)
{
  m_singleASCIIMask = newMask;
}

G4bool GateSingleDigi::GetSingleASCIIMask(G4int index)
{
  G4bool mask = m_singleASCIIMaskDefault;
  if ((index >=0 ) && (((unsigned int)index) < m_singleASCIIMask.size())) mask = m_singleASCIIMask[index];
  return mask;
}

std::vector<G4bool> GateSingleDigi::GetSingleASCIIMask()
{
  return m_singleASCIIMask;
}
