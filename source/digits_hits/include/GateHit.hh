/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

#ifndef GateHit_h
#define GateHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"

#include "globals.hh"
#include <iostream>
#include <iomanip>
#include <fstream>

#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"

/*! \class  GateHit
    \brief  Stores hit information for a hit taking place in a volume connected to a system

    - GateHit - by Giovanni Santin

*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.
//    Last modification in 05/2022 by Olga.Kochebina@cea.fr for NewGateDigitizerOld



class GateHit : public G4VHit
{
  public:

      GateHit();
      ~GateHit();
      //GateHit(const GateHit &right);
      //const GateHit& operator=(const GateHit &right);
      //int operator==(const GateHit &right) const;

      inline void *operator new(size_t);
      inline void operator delete(void *aHit);

      void Draw();
      void Print();

      friend std::ostream& operator<<(std::ostream& flux, const GateHit& hit);

      friend std::ofstream& operator<<(std::ofstream& flux, GateHit* hit);

public:
  G4double m_edep;            // energy deposit for the current hit
  G4double m_stepLength;      // length of the step for the current hit
  G4double m_trackLength;      // length of the track 
  G4double m_time;            // time of the current hit
  G4double m_trackLocalTime;  // time of the current track
  G4ThreeVector m_pos;        // position of the current hit
  G4double m_posx;
  G4double m_posy;
  G4double m_posz;
  G4ThreeVector m_momDir;        // momentum Direction of the current hit
  G4ThreeVector m_localPos;   // position of the current hit
  G4String m_process;         // process on the current hit
  G4int m_PDGEncoding;        // G4 PDGEncoding
  G4int m_trackID;            // track ID
  G4int m_parentID;           // parent track ID
  G4int m_sourceID;           // source progressive number
  G4ThreeVector m_sourcePosition; // position of the source (NOT the positron) that generated the hit
  G4int m_photonID;           // photon ID (1 or 2, 0 if not caused by one of the 2 gammas)
  G4int m_nPhantomCompton;    // # of compton processes in the phantom occurred to the photon
  G4int m_nCrystalCompton;    // # of compton processes in the crystal occurred to the photon
  G4int m_nPhantomRayleigh;    // # of Rayleigh processes in the phantom occurred to the photon
  G4int m_nCrystalRayleigh;    // # of Rayleigh processes in the crystal occurred to the photon
  G4String m_comptonVolumeName; // name of the volume of the last (if any) compton scattering
  G4String m_RayleighVolumeName; // name of the volume of the last (if any) Rayleigh scattering
  G4int m_primaryID;          // primary that caused the hit
  G4int m_eventID;            // eventID
  G4int m_runID;              // runID
  GateVolumeID m_volumeID;    // Volume ID in the world volume tree
  G4ThreeVector m_scannerPos; // Position of the scanner
  G4double m_scannerRotAngle; // Rotation angle of the scanner
  GateOutputVolumeID m_outputVolumeID;
  G4int m_systemID;           // system ID in for the multi-system approach

  // To use with GateROOTBasicOutput classes
  G4ThreeVector pos;  // position

  // HDS : Added in order to record septal penetration
  G4int m_nSeptal;


// AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume
  G4String m_Postprocess;         // PostStep process 
  G4double m_energyIniTrack;         // Initial energy of the track
  G4double m_energyFin;         // final energy of the particle
  G4double m_sourceEnergy;//AE
  G4int m_sourcePDG;//AE
  G4int m_nCrystalConv;    // # of pair creation process in the crystal occurred to the photon
  
  //Additional information from ExtendedVSource
  G4int m_sourceType = 0;//0 means 'by default not known'; sourceType says what type of positronium (Ps) is used: pPs, oPs 
  G4int m_decayType = 0;//0 means 'by default not known'; decayType says what type of Ps decay is used: standard (without prompt gamma), deexcitation (with prompt gamma)
  G4int m_gammaType = 0;//0 means 'by default not known'; gammaType says what type of gamma is emitted: annihilation, prompt, other

  public:
      inline void SetEdep(G4double de)          { m_edep = de; }
      inline void AddEdep(G4double de)          { m_edep += de; }
      inline G4double GetEdep() const                { return m_edep; }

      inline void SetStepLength(G4double value) { m_stepLength = value; }
      inline G4double GetStepLength() const          { return m_stepLength; }

      inline void SetTrackLength(G4double value) { m_trackLength = value; }
      inline G4double GetTrackLength() const          { return m_trackLength; }
      
      inline void     SetTrackLocalTime(G4double aTime)    { m_trackLocalTime = aTime; }
      inline G4double GetTrackLocalTime() const                { return m_trackLocalTime; }

      inline void     SetTime(G4double aTime)    { m_time = aTime; }
      inline G4double GetTime() const                { return m_time; }

      inline void  SetGlobalPos(const G4ThreeVector& xyz)    { m_pos = xyz; }
      inline const G4ThreeVector& GetGlobalPos() const            { return m_pos; }


      inline void  SetMomentumDir(const G4ThreeVector& xyz)     { m_momDir = xyz; }
      inline const G4ThreeVector& GetMomentumDir() const             { return m_momDir; }

      inline void  SetLocalPos(const G4ThreeVector& xyz)     { m_localPos = xyz; }
      inline const G4ThreeVector& GetLocalPos() const             { return m_localPos; }


      inline void     SetProcess(G4String proc) { m_process = proc; }
      inline G4String GetProcess() const             { return m_process; }

      inline void  SetPDGEncoding(G4int j)      { m_PDGEncoding = j; }
      inline G4int GetPDGEncoding() const            { return m_PDGEncoding; }

      inline void  SetTrackID(G4int j)          { m_trackID = j; }
      inline G4int GetTrackID() const                { return m_trackID; }

      inline void  SetParentID(G4int j)         { m_parentID = j; }
      inline G4int GetParentID() const               { return m_parentID; }

      inline void  SetSourceID(G4int j)         { m_sourceID = j; }
      inline G4int GetSourceID() const               { return m_sourceID; }

      inline void  SetSourcePosition(const G4ThreeVector& xyz)     { m_sourcePosition = xyz; }
      inline const G4ThreeVector& GetSourcePosition() const        { return m_sourcePosition; }

      inline void  SetPhotonID(G4int j)         { m_photonID = j; }
      inline G4int GetPhotonID() const               {  return m_photonID; }

      inline void  SetNPhantomCompton(G4int j)  { m_nPhantomCompton = j; }
      inline G4int GetNPhantomCompton() const        { return m_nPhantomCompton; }

      inline void  SetNCrystalCompton(G4int j)  { m_nCrystalCompton = j; }
      inline G4int GetNCrystalCompton() const        { return m_nCrystalCompton; }

      inline void  SetNPhantomRayleigh(G4int j)  { m_nPhantomRayleigh = j; }
      inline G4int GetNPhantomRayleigh() const        { return m_nPhantomRayleigh; }

      inline void  SetNCrystalRayleigh(G4int j)  { m_nCrystalRayleigh = j; }
      inline G4int GetNCrystalRayleigh() const        { return m_nCrystalRayleigh; }

      inline void     SetComptonVolumeName(G4String name) { m_comptonVolumeName = name; }
      inline G4String GetComptonVolumeName() const        { return m_comptonVolumeName; }

      inline void     SetRayleighVolumeName(G4String name) { m_RayleighVolumeName = name; }
      inline G4String GetRayleighVolumeName() const        { return m_RayleighVolumeName; }

      inline void  SetPrimaryID(G4int j)        { m_primaryID = j; }
      inline G4int GetPrimaryID() const              { return m_primaryID; }

      inline void  SetEventID(G4int j)          { m_eventID = j; }
      inline G4int GetEventID() const                { return m_eventID; }

      inline void  SetRunID(G4int j)            { m_runID = j; }
      inline G4int GetRunID() const                  { return m_runID; }

      inline void  SetVolumeID(const GateVolumeID& volumeID)            { m_volumeID = volumeID; }
      inline const GateVolumeID& GetVolumeID() const                  	{ return m_volumeID; }

      inline void  SetScannerPos(const G4ThreeVector& xyz)            	{ m_scannerPos = xyz; }
      inline const G4ThreeVector& GetScannerPos() const                   	{ return m_scannerPos; }

      inline void     SetScannerRotAngle(G4double angle)      	        { m_scannerRotAngle = angle; }
      inline G4double GetScannerRotAngle() const                   	      	{ return m_scannerRotAngle; }

      inline void  SetOutputVolumeID(const GateOutputVolumeID& outputVolumeID)  { m_outputVolumeID = outputVolumeID; }
      inline const GateOutputVolumeID& GetOutputVolumeID()  const             	{ return m_outputVolumeID; }
      inline G4int GetComponentID(size_t depth) const    { return (m_outputVolumeID.size()>depth) ? m_outputVolumeID[depth] : -1; }

      inline void  SetSystemID(const G4int systemID) { m_systemID = systemID; }
      inline G4int GetSystemID() const { return m_systemID; }

      inline G4bool GoodForAnalysis() const
      	  { return ( (m_process != "Transportation") || (m_edep!=0.) ); }

      // HDS : Added in order to record septal penetration
      inline void  SetNSeptal(G4int j)  { m_nSeptal = j; }
      inline G4int GetNSeptal() const        { return m_nSeptal; }




      // AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume 
      inline void     SetPostStepProcess(G4String proc) { m_Postprocess = proc; }
      inline G4String GetPostStepProcess() const             { return m_Postprocess; }
     
      inline void SetEnergyIniTrack(G4double eIni)          { m_energyIniTrack = eIni; }
      inline G4double GetEnergyIniTrack() const                { return m_energyIniTrack; }

      inline void SetEnergyFin(G4double eFin)          { m_energyFin = eFin; }
      inline G4double GetEnergyFin() const                { return m_energyFin; }

      /////--------------------------------------------------------------


      //------AE-------------------------------------------------------------------------
      inline void  SetSourceEnergy(G4double value)     { m_sourceEnergy = value; }
      inline  G4double GetSourceEnergy() const        { return m_sourceEnergy; }

      inline void  SetSourcePDG(G4int value)     { m_sourcePDG = value; }
      inline  G4int GetSourcePDG() const        { return m_sourcePDG; }


      inline void  SetNCrystalConv(G4int value)     { m_nCrystalConv = value; }
      inline  G4int GetNCrystalConv() const        { return m_nCrystalConv; }
      //----------------------------------------------------------------


      // To test move part of the code ---------------------------------------------------------
      inline void  SetXPos(const G4double & x)    { m_posx = x; }
      inline const G4double& GetXPos() const      { return m_posx; }

      inline void  SetYPos(const G4double & y)    { m_posy = y; }
      inline const G4double& GetYPos() const      { return m_posy; }


      inline void  SetZPos(const G4double & z)    { m_posz = z; }
      inline const G4double& GetZPos() const      { return m_posz; }
      /////--------------------------------------------------------------
      inline void SetSourceType(G4int value) { m_sourceType = value; }
      inline G4int GetSourceType() const { return m_sourceType; }
      
      inline void SetDecayType(G4int value) { m_decayType = value; }
      inline G4int GetDecayType() const { return m_decayType; }
      
      inline void SetGammaType(G4int value){ m_gammaType = value; }
      inline G4int GetGammaType() const { return m_gammaType; }
};


typedef G4THitsCollection<GateHit> GateHitsCollection;

extern G4Allocator<GateHit> GateHitAllocator;

inline void* GateHit::operator new(size_t)
{
  void *aHit;
  aHit = (void *) GateHitAllocator.MallocSingle();
  return aHit;
}

inline void GateHit::operator delete(void *aHit)
{
  GateHitAllocator.FreeSingle((GateHit*) aHit);
}

#endif
