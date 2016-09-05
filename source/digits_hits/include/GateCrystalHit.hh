/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCrystalHit_h
#define GateCrystalHit_h 1

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

/*! \class  GateCrystalHit
    \brief  Stores hit information for a hit taking place in a volume connected to a system

    - GateCrystalHit - by Giovanni Santin

*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateCrystalHit : public G4VHit
{
  public:

      GateCrystalHit();
      ~GateCrystalHit();
      //GateCrystalHit(const GateCrystalHit &right);
      //const GateCrystalHit& operator=(const GateCrystalHit &right);
      //int operator==(const GateCrystalHit &right) const;


      inline void *operator new(size_t);
      inline void operator delete(void *aHit);

      void Draw();
      void Print();

      friend std::ostream& operator<<(std::ostream& flux, const GateCrystalHit& hit);

      friend std::ofstream& operator<<(std::ofstream& flux, GateCrystalHit* hit);

private:
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


      // To test move part of the code ---------------------------------------------------------
      inline void  SetXPos(const G4double & x)    { m_posx = x; }
      inline const G4double& GetXPos() const      { return m_posx; }

      inline void  SetYPos(const G4double & y)    { m_posy = y; }
      inline const G4double& GetYPos() const      { return m_posy; }


      inline void  SetZPos(const G4double & z)    { m_posz = z; }
      inline const G4double& GetZPos() const      { return m_posz; }
      /////--------------------------------------------------------------

};

typedef G4THitsCollection<GateCrystalHit> GateCrystalHitsCollection;

extern G4Allocator<GateCrystalHit> GateCrystalHitAllocator;

inline void* GateCrystalHit::operator new(size_t)
{
  void *aHit;
  aHit = (void *) GateCrystalHitAllocator.MallocSingle();
  return aHit;
}

inline void GateCrystalHit::operator delete(void *aHit)
{
  GateCrystalHitAllocator.FreeSingle((GateCrystalHit*) aHit);
}

#endif
