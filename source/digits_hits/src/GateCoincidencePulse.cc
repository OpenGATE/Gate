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
//    m_Time=src.m_Time;
    m_startTime=src.m_startTime;
    m_offsetWindow=src.m_offsetWindow;
    m_coincidenceWindow=src.m_coincidenceWindow;
}
void GateCoincidencePulse::push_back(GatePulse* newPulse)
{
    if ( newPulse->GetTime() < m_startTime)
      m_startTime = newPulse->GetTime();
//     if ( newPulse->GetTime() > m_Time)
//       m_Time = newPulse->GetTime();
    GatePulseList::push_back(newPulse);
}


std::ostream& operator<<(std::ostream& flux, const GateCoincidencePulse& pulse)
{
  flux    << "----GateCoincidencePulse----"   	      	      	      	       << G4endl
      	  << "\tStart  " << G4BestUnit(pulse.m_startTime,"Time") 	       << G4endl
      	  << "\tOffset  "<< G4BestUnit(pulse.m_offsetWindow,"Time") 	       << G4endl
	  << "\tWindow " << G4BestUnit(pulse.m_coincidenceWindow,"Time")       << G4endl ;
  for (size_t i=0; i<pulse.size(); i++)
     flux << *(pulse[i]) << "\n";
  flux    << "----------------------------"   	      	      	      	       << G4endl;

  return flux;
}

inline G4bool GateCoincidencePulse::IsInCoincidence(const GatePulse* newPulse) const
{
   return    ( size()==0 )
           ||
	     (  (newPulse->GetTime()>=m_startTime+m_offsetWindow)
	       &&
	        (newPulse->GetTime()< (m_startTime+m_offsetWindow+m_coincidenceWindow))
	     );
}
inline G4bool GateCoincidencePulse::IsAfterWindow(const GatePulse* newPulse) const
{
   return (newPulse->GetTime()>=m_startTime+m_offsetWindow+m_coincidenceWindow);
}
void GateCoincidencePulse::InsertUniqueSortedCopy(GatePulse* newPulse)
{
    GatePulseList::InsertUniqueSortedCopy(newPulse);
    if ( newPulse->GetTime() < m_startTime)
      m_startTime = newPulse->GetTime();
//     if ( newPulse->GetTime() > m_Time)
//       m_Time = newPulse->GetTime();
}
