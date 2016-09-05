/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidencePulse.hh"

#include "G4UnitsTable.hh"

GateCoincidencePulse::GateCoincidencePulse(const GateCoincidencePulse& src)
   :GatePulseList(src)
{
    m_startTime = src.m_startTime;
    m_endTime = src.m_endTime;
    m_delayed = src.m_delayed;
}

std::ostream& operator<<(std::ostream& flux, const GateCoincidencePulse& pulse)
{
  flux    << "----GateCoincidencePulse----" << Gateendl
      	  << "\tStart  " << G4BestUnit(pulse.m_startTime,"Time") << Gateendl
      	  << "\tEnd  "<< G4BestUnit(pulse.m_endTime,"Time") << Gateendl;
  for (size_t i=0; i<pulse.size(); i++)
     flux << *(pulse[i]) << "\n";
  flux    << "----------------------------"   	      	      	      	       << Gateendl;

  return flux;
}

inline G4bool GateCoincidencePulse::IsInCoincidence(const GatePulse* newPulse) const
{
   return  ( (newPulse->GetTime() >= m_startTime) && (newPulse->GetTime() < m_endTime) );
}
inline G4bool GateCoincidencePulse::IsAfterWindow(const GatePulse* newPulse) const
{
   return (newPulse->GetTime() >= m_endTime);
}

