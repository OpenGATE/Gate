/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCoincidenceDigi.hh"

#include "G4UnitsTable.hh"
#include "G4DigiManager.hh"

std::vector<G4bool> GateCoincidenceDigi::m_coincidenceASCIIMask;
G4bool              GateCoincidenceDigi::m_coincidenceASCIIMaskDefault;


G4Allocator<GateCoincidenceDigi> GateCoincidenceDigiAllocator;



GateCoincidenceDigi::GateCoincidenceDigi(const void* itsMother)
{
}


GateCoincidenceDigi::GateCoincidenceDigi(GateDigi *firstDigi,
        								G4double itsCoincidenceWindow,
										G4double itsOffsetWindow):
   std::vector<GateDigi*>()
{
	push_back(firstDigi);
	m_coincID=-1;
	m_startTime = firstDigi->GetTime() + itsOffsetWindow;
	m_endTime = m_startTime + itsCoincidenceWindow;
	if(itsOffsetWindow > 0.0)
		m_delayed = true;
	else
		m_delayed = false;


}




GateCoincidenceDigi::GateCoincidenceDigi(const GateCoincidenceDigi& src):
		std::vector<GateDigi*>(src)
{
    m_startTime = src.m_startTime;
    m_endTime = src.m_endTime;
    m_delayed = src.m_delayed;
    m_coincID=src.m_coincID;

}


void GateCoincidenceDigi::Draw()
{;}





void GateCoincidenceDigi::Print()
{

  G4cout << this << Gateendl;

}



std::ofstream& operator<<(std::ofstream& flux, GateCoincidenceDigi* digi)
{
	  GateDigi digi_tmp;
	  for (G4int iP=0; iP<2; iP++) {
	    digi_tmp = digi->GetDigi(iP);
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(0) ) flux << " " << std::setw(7) << digi_tmp.GetRunID();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(1) ) flux << " " << std::setw(7) << digi_tmp.GetEventID();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(2) ) flux << " " << std::setw(5) << digi_tmp.GetSourceID();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(3) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetSourcePosition().x()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(4) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetSourcePosition().y()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(5) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetSourcePosition().z()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(6) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(23) << digi_tmp.GetTime()/s;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(7) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetEnergy()/MeV;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(8) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetGlobalPos().x()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(9) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetGlobalPos().y()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(10) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetGlobalPos().z()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(11) ) flux << " " << std::setw(5) << digi_tmp.GetOutputVolumeID();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(12) ) flux << " " << std::setw(5) << digi_tmp.GetNPhantomCompton();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(13) ) flux << " " << std::setw(5) << digi_tmp.GetNCrystalCompton();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(14) ) flux << " " << std::setw(5) << digi_tmp.GetNPhantomRayleigh();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(15) ) flux << " " << std::setw(5) << digi_tmp.GetNCrystalRayleigh();
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(16) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetScannerPos().z()/mm;
	    if ( GateCoincidenceDigi::GetCoincidenceASCIIMask(17) ) flux << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setprecision(3)  << digi_tmp.GetScannerRotAngle()/deg;;
	  }
	  flux << Gateendl;

	  return flux;
	}

std::ostream& operator<<(std::ostream& flux, const GateCoincidenceDigi& digi)
{
  flux    << "----GateCoincidenceDigi----" << Gateendl
      	  << "\tStart  " << G4BestUnit(digi.GetStartTime(),"Time") << Gateendl
      	  << "\tEnd  "<< G4BestUnit(digi.GetEndTime(),"Time") << Gateendl;
  for (size_t i=0; i<digi.size(); i++)
     flux << *(digi[i]) << "\n";
  flux    << "----------------------------"   	      	      	      	       << Gateendl;

  return flux;
}


inline G4bool GateCoincidenceDigi::IsInCoincidence(const GateDigi* newDigi) const
{
   return  ( (newDigi->GetTime() >= m_startTime) && (newDigi->GetTime() < m_endTime) );
}

inline G4bool GateCoincidenceDigi::IsAfterWindow(const GateDigi* newDigi) const
{
	return (newDigi->GetTime() >= m_endTime);
}


GateDigi* GateCoincidenceDigi::GetDigi(G4int i)
{
	return this->at(i);
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

