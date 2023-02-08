/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

//GND 2022 Class to Remove
#include "GateConfiguration.h"
#include "GatePulse.hh"
#include "GateVSystem.hh"

#include "G4UnitsTable.hh"

GatePulse::GatePulse(const void* itsMother)
    : m_runID(-1),
      m_eventID(-1),
      m_sourceID(-1),
      m_time(0),
      m_energy(0),
	  m_max_energy(0),
      m_nPhantomCompton(-1),
      m_nPhantomRayleigh(-1),
      #ifdef GATE_USE_OPTICAL
      m_optical(false),
      #endif
      m_energyError(0.0),
      m_globalPosError(0.0),
      m_localPosError(0.0),
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


    // AE : Added in a real pulse no sense
    m_Postprocess="NULL";         // PostStep process
    m_energyIniTrack=-1;         // Initial energy of the track
    m_energyFin=-1;         // final energy of the particle
    m_processCreator="NULL";
    m_trackID=0;
    //-----------------

    // time: store the minimum time
    m_time = std::min ( m_time , right->m_time ) ;

    // energy: we compute the sum, but do not store it yet
    // (storing it now would mess up the centroid computations)
    G4double totalEnergy = m_energy + right->m_energy;

    if (m_sourceEnergy != right->m_sourceEnergy) m_sourceEnergy=-1;
    if (m_sourcePDG != right->m_sourcePDG) m_sourcePDG=0;
    if ( right->m_nCrystalConv > m_nCrystalConv ){
        m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > m_nCrystalCompton ){
        m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > m_nCrystalRayleigh ){
        m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }

    // Local and global positions: store the controids
    if(totalEnergy>0){
        m_localPos  =  ( m_localPos  * m_energy  + right->m_localPos  * right->m_energy ) / totalEnergy ;
        m_globalPos =  ( m_globalPos * m_energy  + right->m_globalPos * right->m_energy ) / totalEnergy ;
    }
    else{
        m_localPos  =  ( m_localPos  + right->m_localPos)/2;
        m_globalPos =  ( m_globalPos + right->m_globalPos)/2 ;
    }

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

const GatePulse & GatePulse::MergePositionEnergyWin(const GatePulse* right){


    // AE : Added in a real pulse no sense
    m_Postprocess="NULL";         // PostStep process
    m_energyIniTrack=0;         // Initial energy of the track
    m_energyFin=0;         // final energy of the particle
    m_processCreator="NULL";
    m_trackID=0;
    //-----------------

    // time: store the minimum time
    m_time = std::min ( m_time , right->m_time ) ;
    if (m_sourceEnergy != right->m_sourceEnergy) m_sourceEnergy=-1;
    if (m_sourcePDG != right->m_sourcePDG) m_sourcePDG=0;
    if ( right->m_nCrystalConv > m_nCrystalConv ){
        m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > m_nCrystalCompton ){
        m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > m_nCrystalRayleigh ){
        m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }



    if( right->m_energy>m_max_energy){
    	m_max_energy=right->m_energy;
        // Local and global positions: store the controids
        m_localPos  =   right->m_localPos;
        m_globalPos =   right->m_globalPos;

    }

    m_energy = m_energy + right->m_energy;


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

const GatePulse& GatePulse::CentroidMergeComptPhotIdeal(const GatePulse* right)
{
    // We define below the fields of the merged pulse

    // runID: identical for both pulses, nothing to do
    // eventID: identical for both pulses, nothing to do
    // sourceID: identical for both pulses, nothing to do
    // source-position: identical for both pulses, nothing to do

    // time: store the minimum time
    m_time = std::min ( m_time , right->m_time ) ;

    if (m_sourceEnergy != right->m_sourceEnergy) m_sourceEnergy=-1;
    if (m_sourcePDG != right->m_sourcePDG) m_sourcePDG=0;
    if ( right->m_nCrystalConv > m_nCrystalConv ){
        m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > m_nCrystalCompton ){
        m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > m_nCrystalRayleigh ){
        m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }
    // energy: we compute the sum
    G4double totalEnergy = m_energy + right->m_energy;

    // Local and global positions: keep the original Position

    // n store the energy
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

    if (m_sourceEnergy != right->m_sourceEnergy) m_sourceEnergy=-1;
    if (m_sourcePDG != right->m_sourcePDG) m_sourcePDG=0;
    if ( right->m_nCrystalConv > m_nCrystalConv ){
        m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > m_nCrystalCompton ){
        m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > m_nCrystalRayleigh ){
        m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }
    m_energyIniTrack=-1;         // Initial energy of the track
    m_energyFin=-1;

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

// Change a volume selector inside the volumeID. Meaning we replace at this depth the volume by the given copyNo.
// The given copyNo is intended to reflect the crystal_id inside the crystal component, not obviously the daughter
// id which can be different if the level above the crystal has daughters declared before the crystal.
// Currently, this method is used in the readout digitizer module for the energy centroid policy where we compute
// a new crystal_id inside the crystal component.
// This method also update the associated outputVolumeID
void GatePulse::ChangeVolumeIDAndOutputVolumeIDValue(size_t depth, G4int copyNo)
{
    if (depth==0) return;
    // Note: The given depth corresponds to the crystal component inside the system.
    //       So in the outputVolumeID this really corresponds to the crystal.
    //       But in the volumeID, this actually corresponds to the level above.

    // Get the old crystal_id of the pulse at the crystal level
    G4int old_crystal_id = m_outputVolumeID[depth];
    // Get the id of this crystal corresponding to the actual daughter number.
    // (it won't be the same if the level above the crystal has daughters declared before the crystal component)
    G4int old_crystal_daughter_id = m_volumeID[depth+1].GetDaughterID();
    // The daughter id is thus higher or equal to the crystal id.
    // We can thus deduce the shift to be applied to the given copyNo in parameter
    G4int shift_id = old_crystal_daughter_id - old_crystal_id;
    // Get physical volume above the given depth which corresponds to the crystal depth inside the system.
    // But for the the volumeID, we must add 1 as the world is the first volume in the vector.
    G4VPhysicalVolume* phys_vol = m_volumeID[depth].GetVolume();
    // Get physical volume of the wanted copy
    phys_vol = phys_vol->GetLogicalVolume()->GetDaughter(copyNo+shift_id);
    // Create a volume selector with this volume
    GateVolumeSelector* volSelector = new GateVolumeSelector(phys_vol);
    // Copy the content of this selector into the given depth
    m_volumeID[depth+1] = *volSelector;
    // Delete the temporary volume selector
    delete volSelector;
    // Finally change the outputVolumeID accordingly
    m_outputVolumeID[depth] = copyNo;
}

// Reset the global position of the pulse with respect to its volumeID that has been changed previously
void GatePulse::ResetGlobalPos(GateVSystem* system)
{
    SetGlobalPos(system->ComputeObjectCenter(&m_volumeID));
}


std::ostream& operator<<(std::ostream& flux, const GatePulse& pulse)
{
    flux    << "\t----GatePulse----"     	      	      	      	      	      	      	      	      	      	               << Gateendl
            << "\t\t" << "Run           " << pulse.m_runID                      	         	      	      	      	       << Gateendl
            << "\t\t" << "Event         " << pulse.m_eventID   	      	      	      	              	      	      	       << Gateendl
            << "\t\t" << "Src           " << pulse.m_sourceID << " [ " << G4BestUnit(pulse.m_sourcePosition,"Length")     << "]\n"
            << "\t\t" << "Time          " << G4BestUnit(pulse.m_time,"Time")      	      	      	      	      	       << Gateendl
            << "\t\t" << "Energy        " << G4BestUnit(pulse.m_energy,"Energy")          	      	      	      	       << Gateendl
            << "\t\t" << "localPos      [ " << G4BestUnit(pulse.m_localPos,"Length")        	      	      	      	<< "]\n"
            << "\t\t" << "globalPos     [ " << G4BestUnit(pulse.m_globalPos,"Length")   	      	      	      	      	<< "]\n"
            << "\t\t" << "           -> ( R="   << G4BestUnit(pulse.m_globalPos.perp(),"Length")     << ", "
            << "phi="   << pulse.m_globalPos.phi()/degree       	      	   << " deg,"
            << "z="     << G4BestUnit(pulse.m_globalPos.z(),"Length")     	     	      	<< ")\n"
            << "\t\t" << "VolumeID      " << pulse.m_volumeID      	      	      	      	      	      	               << Gateendl
            << "\t\t" << "OutputID      " << pulse.m_outputVolumeID     	      	      	      	      	      	      	       << Gateendl
            << "\t\t" << "#Compton      " << pulse.m_nPhantomCompton      	      	      	      	      	      	       << Gateendl
            << "\t\t" << "#Rayleigh     " << pulse.m_nPhantomRayleigh      	      	      	      	      	      	       << Gateendl
            << "\t\t" << "scannerPos    [ " << G4BestUnit(pulse.m_scannerPos,"Length")        	      	      	      	<< "]\n" << Gateendl
            << "\t\t" << "scannerRotAngle " << pulse.m_scannerRotAngle/degree           	      	      	      	     << " deg\n" << Gateendl
            << "\t-----------------\n";

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
        pop_back();
        // erase(end()-1);
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
    for ( const_iterator iter = begin(); iter < end() ; ++iter)
        ans += (*iter)->GetEnergy();

    return ans;
}
//Insert a new pulse in the good place wrt it time of arrival
void GatePulseList::InsertUniqueSortedCopy(GatePulse* newPulse)
{
    GatePulseIterator it;
    for (it=this->begin();it!=this->end();it++){
        if ( (*it)->GetTime()==newPulse->GetTime() ) return;
        if ( (*it)->GetTime()>newPulse->GetTime()){
            this->insert(it,new GatePulse(*newPulse));
            break;
        }
    }
    if (it==this->end()) push_back(new GatePulse(*newPulse));
}

