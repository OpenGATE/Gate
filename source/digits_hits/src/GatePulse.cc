/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#include "GatePulse.hh"

#include "G4UnitsTable.hh"

GatePulse::GatePulse(const void* itsMother)
  : m_runID(-1),
    m_eventID(-1),
    m_sourceID(-1),
    m_time(0),
    m_energy(0),
    m_nPhantomCompton(-1),
    m_nPhantomRayleigh(-1),
#ifdef GATE_USE_OPTICAL
    m_optical(false),
#endif
    m_mother(itsMother)
{
}

const GatePulse& GatePulse::CentroidMerge(const GatePulse* right)
{
  // We define below the fields of the merged pulse

  // runID: identical for both pulses, nothing to do
  // eventID: identical for both pulses, nothing to do
  // sourceID: identical for both pulses, nothing to do
  // source-position: identical for both pulses, nothing to do

  // time: store the minimum time
  m_time = std::min ( m_time , right->m_time ) ;

  // energy: we compute the sum, but do not store it yet
  // (storing it now would mess up the centroid computations)
  G4double totalEnergy = m_energy + right->m_energy;

  // Local and global positions: store the controids
  m_localPos  =  ( m_localPos  * m_energy  + right->m_localPos  * right->m_energy ) / totalEnergy ;
  m_globalPos =  ( m_globalPos * m_energy  + right->m_globalPos * right->m_energy ) / totalEnergy ;

  // Now that the centroids are stored, we can store the energy
  m_energy   = totalEnergy;

  // # of compton process: store the max nb
  if ( right->m_nPhantomCompton > m_nPhantomCompton )
  {
    m_nPhantomCompton 	= right->m_nPhantomCompton;
    m_comptonVolumeName = right->m_comptonVolumeName;
  }

  // # of Rayleigh process: store the max nb
  if ( right->m_nPhantomRayleigh > m_nPhantomRayleigh )
  {
    m_nPhantomRayleigh 	= right->m_nPhantomRayleigh;
    m_RayleighVolumeName = right->m_RayleighVolumeName;
  }

  // HDS : # of septal hits: store the max nb
	if ( right->m_nSeptal > m_nSeptal )
	{
		m_nSeptal 	= right->m_nSeptal;
	}

  // VolumeID: should be identical for both pulses, we do nothing
  // m_scannerPos: identical for both pulses, nothing to do
  // m_scannerRotAngle: identical for both pulses, nothing to do
  // m_outputVolumeID: should be identical for both pulses, we do nothing

  return *this;
}

const GatePulse& GatePulse::CentroidMergeCompton(const GatePulse* right)
{
  // We define below the fields of the merged pulse

  // runID: identical for both pulses, nothing to do
  // eventID: identical for both pulses, nothing to do
  // sourceID: identical for both pulses, nothing to do
  // source-position: identical for both pulses, nothing to do

  // time: store the minimum time
  m_time = std::min ( m_time , right->m_time ) ;

  // energy: we compute the sum
  G4double totalEnergy = m_energy + right->m_energy;

  // Local and global positions: keep the original Position

  // Now that the centroids are stored, we can store the energy
  m_energy   = totalEnergy;

  // # of compton process: store the max nb
  if ( right->m_nPhantomCompton > m_nPhantomCompton )
  {
    m_nPhantomCompton 	= right->m_nPhantomCompton;
    m_comptonVolumeName = right->m_comptonVolumeName;
  }

  // # of Rayleigh process: store the max nb
  if ( right->m_nPhantomRayleigh > m_nPhantomRayleigh )
  {
    m_nPhantomRayleigh 	= right->m_nPhantomRayleigh;
    m_RayleighVolumeName = right->m_RayleighVolumeName;
  }

    // HDS : # of septal hits: store the max nb
	if ( right->m_nSeptal > m_nSeptal )
	{
		m_nSeptal 	= right->m_nSeptal;
	}

  // VolumeID: should be identical for both pulses, we do nothing
  // m_scannerPos: identical for both pulses, nothing to do
  // m_scannerRotAngle: identical for both pulses, nothing to do
  // m_outputVolumeID: should be identical for both pulses, we do nothing

  return *this;
}

std::ostream& operator<<(std::ostream& flux, const GatePulse& pulse)
{
  flux    << "\t----GatePulse----"     	      	      	      	      	      	      	      	      	      	               << G4endl
	  << "\t\t" << "Run           " << pulse.m_runID                      	         	      	      	      	       << G4endl
	  << "\t\t" << "Event         " << pulse.m_eventID   	      	      	      	              	      	      	       << G4endl
	  << "\t\t" << "Src           " << pulse.m_sourceID << " [ " << G4BestUnit(pulse.m_sourcePosition,"Length")     << "]" << G4endl
      	  << "\t\t" << "Time          " << G4BestUnit(pulse.m_time,"Time")      	      	      	      	      	       << G4endl
      	  << "\t\t" << "Energy        " << G4BestUnit(pulse.m_energy,"Energy")          	      	      	      	       << G4endl
      	  << "\t\t" << "localPos      [ " << G4BestUnit(pulse.m_localPos,"Length")        	      	      	      	<< "]" << G4endl
      	  << "\t\t" << "globalPos     [ " << G4BestUnit(pulse.m_globalPos,"Length")   	      	      	      	      	<< "]" << G4endl
       	  << "\t\t" << "           -> ( R="   << G4BestUnit(pulse.m_globalPos.perp(),"Length")     << ", "
	      	      	      	  << "phi="   << pulse.m_globalPos.phi()/degree       	      	   << " deg,"
				  << "z="     << G4BestUnit(pulse.m_globalPos.z(),"Length")     	     	      	<< ")" << G4endl
 	  << "\t\t" << "VolumeID      " << pulse.m_volumeID      	      	      	      	      	      	               << G4endl
	  << "\t\t" << "OutputID      " << pulse.m_outputVolumeID     	      	      	      	      	      	      	       << G4endl
	  << "\t\t" << "#Compton      " << pulse.m_nPhantomCompton      	      	      	      	      	      	       << G4endl
	  << "\t\t" << "#Rayleigh     " << pulse.m_nPhantomRayleigh      	      	      	      	      	      	       << G4endl
      	  << "\t\t" << "scannerPos    [ " << G4BestUnit(pulse.m_scannerPos,"Length")        	      	      	      	<< "]" << G4endl
      	  << "\t\t" << "scannerRotAngle " << pulse.m_scannerRotAngle/degree           	      	      	      	     << " deg" << G4endl
     	  << "\t-----------------" << G4endl;

  return flux;
}


// Copy constructor : make new copies of pulses
GatePulseList::GatePulseList(const GatePulseList& src):
std::vector<GatePulse*>()
{
    m_name=src.m_name;
    for (GatePulseConstIterator it=src.begin();it != src.end() ;++it)
    	push_back(new GatePulse( **it ));
}
GatePulseList::~GatePulseList()
{
  while (!empty()) {
        delete back();
        erase(end()-1);
  }
}





// Return the min-time of all pulses
GatePulse* GatePulseList::FindFirstPulse() const
{
  G4double startTime = DBL_MAX;
  GatePulse* ans=0;
  for ( const_iterator iter = begin(); iter < end() ; ++iter) {
      if ( (*iter)->GetTime() < startTime ){
      	startTime  = (*iter)->GetTime();
	ans = *iter;
      }
  }

 return ans;
}
// Return the min-time of all pulses
G4double GatePulseList::ComputeStartTime() const
{
    GatePulse* pulse = FindFirstPulse();
    return pulse? pulse->GetTime() : DBL_MAX;
}

// Return the max-time of all pulses
G4double GatePulseList::ComputeFinishTime() const
{
  G4double finishTime = 0;
  for ( const_iterator iter = begin(); iter < end() ; ++iter)
      if ( (*iter)->GetTime() > finishTime )
      	finishTime  = (*iter)->GetTime();

 return finishTime;
}
// Return the total energy of the pulse list
G4double GatePulseList::ComputeEnergy() const
{
  G4double ans = 0;
  for ( const_iterator iter = begin(); iter < end() ; ++iter) ans += (*iter)->GetEnergy();

 return ans;
}
//Insert a new pulse in the good place wrt it time of arrival
void GatePulseList::InsertUniqueSortedCopy(GatePulse* newPulse)
{
    GatePulseList::iterator it;
    for (it=this->begin();it!=this->end();it++){
    	if ( (*it)->GetTime()==newPulse->GetTime() ) return;
    	if ( (*it)->GetTime()>newPulse->GetTime()){
	    this->insert(it,new GatePulse(*newPulse));
	    break;
	}
    }
    if (it==this->end()) push_back(new GatePulse(*newPulse));
}
