/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

// OK GND 2022
/*!
  \class  GateDigi
  \brief  New class that regroup the two old classes: GatePulse and GateDigi

    - GateDigi is an object that is used to construct Singles (i.e. digitized Hits)
    - They are stored in GateDigiCollections manages by G4DigiMan and GateDigitizerMgr
	- GateDigi is obtained from Hits with GateDigitizerInitializationModule

*/

#include "GateDigi.hh"
#include "G4UnitsTable.hh"

#include <iomanip>

std::vector<G4bool> GateDigi::m_singleASCIIMask;
G4bool                GateDigi::m_singleASCIIMaskDefault;

G4Allocator<GateDigi> GateDigiAllocator;

GateDigi::GateDigi(const void* itsMother):
	  m_runID(-1),
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

void GateDigi::Draw()
{;}





void GateDigi::Print()
{

  G4cout << this << Gateendl;

}



std::ostream& operator<<(std::ostream& flux, const GateDigi& digi)
{
      flux    << "\t----GateDigi----"     	      	      	      	      	      	      	      	      	      	               << Gateendl
              << "\t\t" << "Run           " << digi.m_runID                      	         	      	      	      	       << Gateendl
              << "\t\t" << "Event         " << digi.m_eventID   	      	      	      	              	      	      	       << Gateendl
              << "\t\t" << "Src           " << digi.m_sourceID << " [ " << G4BestUnit(digi.m_sourcePosition,"Length")     << "]\n"
              << "\t\t" << "Time          " << G4BestUnit(digi.m_time,"Time")      	      	      	      	      	       << Gateendl
              << "\t\t" << "Energy        " << G4BestUnit(digi.m_energy,"Energy")          	      	      	      	       << Gateendl
              << "\t\t" << "localPos      [ " << G4BestUnit(digi.m_localPos,"Length")        	      	      	      	<< "]\n"
              << "\t\t" << "globalPos     [ " << G4BestUnit(digi.m_globalPos,"Length")   	      	      	      	      	<< "]\n"
              << "\t\t" << "           -> ( R="   << G4BestUnit(digi.m_globalPos.perp(),"Length")     << ", "
              << "phi="   << digi.m_globalPos.phi()/degree       	      	   << " deg,"
              << "z="     << G4BestUnit(digi.m_globalPos.z(),"Length")     	     	      	<< ")\n"
              << "\t\t" << "VolumeID      " << digi.m_volumeID      	      	      	      	      	      	               << Gateendl
              << "\t\t" << "OutputID      " << digi.m_outputVolumeID     	      	      	      	      	      	      	       << Gateendl
              << "\t\t" << "#Compton      " << digi.m_nPhantomCompton      	      	      	      	      	      	       << Gateendl
              << "\t\t" << "#Rayleigh     " << digi.m_nPhantomRayleigh      	      	      	      	      	      	       << Gateendl
              << "\t\t" << "scannerPos    [ " << G4BestUnit(digi.m_scannerPos,"Length")        	      	      	      	<< "]\n" << Gateendl
              << "\t\t" << "scannerRotAngle " << digi.m_scannerRotAngle/degree           	      	      	      	     << " deg\n" << Gateendl
              << "\t-----------------\n";

  return flux;
}

std::ofstream& operator<<(std::ofstream& flux, GateDigi* digi)
{
  flux << " " << std::setw(7) << digi->GetRunID()
    << " " << std::setw(7) << digi->GetEventID()
    << " " << std::setw(5) << digi->GetSourceID()
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3) << digi->GetSourcePosition().x()/mm
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3) << digi->GetSourcePosition().y()/mm
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3) << digi->GetSourcePosition().z()/mm
    << " " << std::setw(5) << digi->GetOutputVolumeID()
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(30) << std::setprecision(23) << digi->GetTime()/s
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetEnergy()/MeV
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetGlobalPos().x()/mm
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetGlobalPos().y()/mm
    << " " << std::resetiosflags(std::ios::floatfield) << std::setiosflags(std::ios::scientific) << std::setw(10) << std::setprecision(3)  << digi->GetGlobalPos().z()/mm
    << " " << std::setw(4) << digi->GetNPhantomCompton()
    << " " << std::setw(4) << digi->GetNCrystalCompton()
    << " " << std::setw(4) << digi->GetNPhantomRayleigh()
    << " " << std::setw(4) << digi->GetNCrystalRayleigh()
    << " " << digi->GetComptonVolumeName()
    << " " << digi->GetRayleighVolumeName()
    << Gateendl;

  return flux;
}


void GateDigi::SetSingleASCIIMask(G4bool newValue)
{
  m_singleASCIIMaskDefault = newValue;
  for (G4int iMask=0; ((unsigned int)iMask)<m_singleASCIIMask.size(); iMask++) {
    m_singleASCIIMask[iMask] = newValue;
  }
}


void GateDigi::SetSingleASCIIMask(std::vector<G4bool> newMask)
{
  m_singleASCIIMask = newMask;
}

G4bool GateDigi::GetSingleASCIIMask(G4int index)
{
  G4bool mask = m_singleASCIIMaskDefault;
  if ((index >=0 ) && (((unsigned int)index) < m_singleASCIIMask.size())) mask = m_singleASCIIMask[index];
  return mask;
}

std::vector<G4bool> GateDigi::GetSingleASCIIMask()
{
  return m_singleASCIIMask;
}


// Change a volume selector inside the volumeID. Meaning we replace at this depth the volume by the given copyNo.
// The given copyNo is intended to reflect the crystal_id inside the crystal component, not obviously the daughter
// id which can be different if the level above the crystal has daughters declared before the crystal.
// Currently, this method is used in the readout digitizer module for the energy centroid policy where we compute
// a new crystal_id inside the crystal component.
// This method also update the associated outputVolumeID
void GateDigi::ChangeVolumeIDAndOutputVolumeIDValue(size_t depth, G4int copyNo)
{
	//G4cout<<"GateDigi::ChangeVolumeIDAndOutputVolumeIDValue"<<G4endl;
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
    //phys_vol = phys_vol->GetLogicalVolume()->GetDaughter(copyNo+shift_id);
    //G4cout<<"phys_vol "<< phys_vol->GetName() <<G4endl;
    if(phys_vol->GetLogicalVolume()->GetNoDaughters()!=0)
       {
       	phys_vol = phys_vol->GetLogicalVolume()->GetDaughter(copyNo+shift_id);
       }
       else
       {
       	GateError("Error: not all required geometry levels and sublevels for this system are defined. "
       			  			  "(Ex.: for cylindricalPET, the required levels are: rsector, module, submodule, crystal). Please, add them to your geometry macro for correct execution of the Readout module");
       }

    // Create a volume selector with this volume
    //G4cout<<"depth "<< depth <<" "<< copyNo <<" "<<shift_id << G4endl;
    //G4cout<<"phys_vol "<< phys_vol->GetName() <<G4endl;

    GateVolumeSelector* volSelector = new GateVolumeSelector(phys_vol);
    // Copy the content of this selector into the given depth
    //G4cout<<"m_volumeID size "<< m_volumeID.size()<<G4endl;
    //G4cout<<"depth+1 "<< depth+1<<G4endl;
    m_volumeID[depth+1] = *volSelector;
    // Delete the temporary volume selector
    delete volSelector;
    // Finally change the outputVolumeID accordingly
    m_outputVolumeID[depth] = copyNo;

    //With the new method the line above could be changed to:
    //SetOutputVolumeID(copyNo,depth)
}





