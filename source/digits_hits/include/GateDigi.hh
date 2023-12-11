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
	05/2022 Olga.Kochebina@cea.fr
*/


#ifndef GateDigi_h
#define GateDigi_h 1

#include "G4VDigi.hh"
#include "G4TDigiCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include <fstream>
#include <iterator>

#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateVSystem.hh"


class GateDigi : public G4VDigi
{

public:

  GateDigi(const void* itsMother=0);

  //! Constructor that takes a pointer!
  inline GateDigi(const GateDigi* right)
     {
         *this = *right;
     }


  virtual inline ~GateDigi() {}

  inline void* operator new(size_t);
  inline void  operator delete(void*);

  void Draw();
  void Print();

  //
  //printing methods
  //
  friend std::ostream& operator<<(std::ostream&, const GateDigi& );

  friend std::ofstream& operator<<(std::ofstream&, GateDigi* );


public:
  	  inline void  SetMother(const void* mother)      	      { m_mother = mother; }
      inline const void* GetMother()   const                  { return m_mother; }

      inline void  SetRunID(G4int j)                  	      { m_runID = j; }
      inline G4int GetRunID()  const                       	      { return m_runID; }

      inline void  SetEventID(G4int j)                	      { m_eventID = j; }
      inline G4int GetEventID() const                      	      { return m_eventID; }

      inline void  SetSourceID(G4int j)               	      { m_sourceID = j; }
      inline G4int GetSourceID()  const                       { return m_sourceID; }

      inline void  SetSourcePosition(const G4ThreeVector& xyz)	{ m_sourcePosition = xyz; }
      inline const G4ThreeVector& GetSourcePosition() const          { return m_sourcePosition; }

      inline G4double GetTime() const                         { return m_time; }
      inline void     SetTime(G4double value)         	      { m_time = value; }

      inline G4double GetEnergy()   const                  	      { return m_energy; }
      inline void SetEnergy(G4double value)           	      { m_energy = value; }

      inline G4double GetMaxEnergy()   const                  	      { return m_max_energy; }
      inline void SetMaxEnergy(G4double value)           	      { m_max_energy = value; }


      inline void  SetPDGEncoding(const G4int j)						{ m_PDGEncoding = j; }
      inline G4int GetPDGEncoding() const						{ return m_PDGEncoding; }

      inline void  SetLocalPos(const G4ThreeVector& xyz)      { m_localPos = xyz; }
      inline const G4ThreeVector& GetLocalPos() const              { return m_localPos; }

      inline void  SetGlobalPos(const G4ThreeVector& xyz)     { m_globalPos = xyz; }
      inline const G4ThreeVector& GetGlobalPos()  const            { return m_globalPos; }

      inline void  SetNPhantomCompton(G4int j)  { m_nPhantomCompton = j; }
      inline G4int GetNPhantomCompton() const        { return m_nPhantomCompton; }

      inline void  SetNCrystalCompton(G4int j)  { m_nCrystalCompton = j; }
      inline G4int GetNCrystalCompton() const        { return m_nCrystalCompton; }

      inline void  SetNPhantomRayleigh(G4int j)  { m_nPhantomRayleigh = j; }
      inline G4int GetNPhantomRayleigh() const        { return m_nPhantomRayleigh; }

      inline void  SetNCrystalRayleigh(G4int j)  { m_nCrystalRayleigh = j; }
      inline G4int GetNCrystalRayleigh() const        { return m_nCrystalRayleigh; }

      inline void     SetComptonVolumeName(const G4String& name) { m_comptonVolumeName = name; }
      inline G4String GetComptonVolumeName() const        { return m_comptonVolumeName; }

      inline void     SetRayleighVolumeName(const G4String& name) { m_RayleighVolumeName = name; }
      inline G4String GetRayleighVolumeName() const        { return m_RayleighVolumeName; }

      inline void  SetVolumeID(const GateVolumeID& volumeID)            { m_volumeID = volumeID; }
      inline const GateVolumeID& GetVolumeID() const                  	{ return m_volumeID; }

      inline void  SetScannerPos(const G4ThreeVector& xyz)            	{ m_scannerPos = xyz; }
      inline const G4ThreeVector& GetScannerPos() const                   	{ return m_scannerPos; }

      inline void     SetScannerRotAngle(G4double angle)      	        { m_scannerRotAngle = angle; }
      inline G4double GetScannerRotAngle() const                   	      	{ return m_scannerRotAngle; }

      inline void  SetOutputVolumeID(const GateOutputVolumeID& outputVolumeID)        	{ m_outputVolumeID = outputVolumeID; }
      inline const GateOutputVolumeID& GetOutputVolumeID()  const             	      	{ return m_outputVolumeID; }
      inline G4int GetComponentID(size_t depth) const    { return (m_outputVolumeID.size()>depth) ? m_outputVolumeID[depth] : -1; }

      inline void  SetSystemID(const G4int systemID) { m_systemID = systemID; }
      inline G4int GetSystemID() const { return m_systemID; }


  #ifdef GATE_USE_OPTICAL
      inline void   SetOptical(G4bool optical = true) { m_optical = optical;}
      inline G4bool IsOptical() const { return m_optical;}
  #endif

      // HDS : record septal penetration
      inline G4int GetNSeptal() const { return m_nSeptal; }
      inline void SetNSeptal(G4int septalNb) { m_nSeptal = septalNb; }




      // AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume
      inline void     SetPostStepProcess(G4String proc) { m_Postprocess = proc; }
      inline G4String GetPostStepProcess() const             { return m_Postprocess; }

      inline void SetEnergyIniTrack(G4double eIni)          { m_energyIniTrack = eIni; }
      inline G4double GetEnergyIniTrack() const                { return m_energyIniTrack; }

      inline void SetEnergyFin(G4double eFin)          { m_energyFin = eFin; }
      inline G4double GetEnergyFin() const                { return m_energyFin; }

      inline void SetSourceEnergy(G4double eValue)          { m_sourceEnergy = eValue; }
      inline G4double GetSourceEnergy() const                { return m_sourceEnergy; }

      inline void SetSourcePDG(G4int PDGEncoding)          { m_sourcePDG = PDGEncoding; }
      inline G4int GetSourcePDG() const                { return m_sourcePDG; }

      inline void SetNCrystalConv(G4int nConv)          { m_nCrystalConv = nConv; }
      inline G4int GetNCrystalConv() const                { return m_nCrystalConv; }


      inline void     SetProcessCreator(G4String proc) { m_processCreator = proc; }
      inline G4String GetProcessCreator() const             { return m_processCreator; }

      inline void SetTrackID(G4int trkID)          { m_trackID = trkID; }
      inline G4int GetTrackID() const                { return m_trackID; }
      inline void SetParentID(G4int parentID)          { m_parentID = parentID; }
      inline G4int GetParentID() const                { return m_parentID; }

      //AE
      inline G4double GetEnergyError()   const                  	      { return m_energyError; }
      inline void SetEnergyError(G4double value)           	      { m_energyError = value; }

      inline void  SetLocalPosError(const G4ThreeVector& xyz)      { m_localPosError = xyz; }
      inline const G4ThreeVector& GetLocalPosError() const              { return m_localPosError; }

      inline void  SetGlobalPosError(const G4ThreeVector& xyz)     { m_globalPosError = xyz; }
      inline const G4ThreeVector& GetGlobalPosError()  const            { return m_globalPosError; }

      //AE for CSR
       bool operator <( const GateDigi& rhs )
       {
           return  GetGlobalPos().getZ() <= rhs.GetGlobalPos().getZ() ;

       }
       //--------------------------------------------------------------------------------

	public:
       static void SetSingleASCIIMask(G4bool);
       static void SetSingleASCIIMask(std::vector<G4bool>);
       static std::vector<G4bool> GetSingleASCIIMask();
       static G4bool GetSingleASCIIMask(G4int index);

       //! S. Stute: Modify one value inside the VolumeID and outputVolumeID vectors
        void ChangeVolumeIDAndOutputVolumeIDValue(size_t depth, G4int value);
	protected:
       static std::vector<G4bool> m_singleASCIIMask;
       static G4bool                m_singleASCIIMaskDefault;


public:
  G4int m_runID;      	      	  //!< runID
  G4int m_eventID;            	  //!< eventID
  G4int m_sourceID;           	  //!< source progressive number
  G4ThreeVector m_sourcePosition; //!< position of the source (NOT the positron) that generated the hit
  G4double m_time;            	  //!< start time of the current pulse
  G4double m_energy;          	  //!< energy measured for the current pulse
  G4double m_max_energy;          	  //!< max energy for the current pulse
  G4ThreeVector m_localPos;   	  //!< position of the current hit
  G4ThreeVector m_globalPos;      //!< position of the current hit
  G4int m_PDGEncoding;        // G4 PDGEncoding
  G4int m_nPhantomCompton;    	  //!< # of compton processes in the phantom occurred to the photon
  G4int m_nCrystalCompton;    	  //!< # of compton processes in the crystal occurred to the photon
  G4int m_nPhantomRayleigh;    	  //!< # of Rayleigh processes in the phantom occurred to the photon
  G4int m_nCrystalRayleigh;    	  //!< # of Rayleigh processes in the crystal occurred to the photon
  G4String m_comptonVolumeName;   //!< name of the volume of the last (if any) compton scattering
  G4String m_RayleighVolumeName;   //!< name of the volume of the last (if any) Rayleigh scattering
  GateVolumeID m_volumeID;        //!< Volume ID in the world volume tree
  G4ThreeVector m_scannerPos; 	  //!< Position of the scanner
  G4double m_scannerRotAngle; 	  //!< Rotation angle of the scanner
  GateOutputVolumeID m_outputVolumeID;
  G4int m_systemID;           // system ID in for the multi-system approach

  #ifdef GATE_USE_OPTICAL
  	  G4bool m_optical;               //!< Is the pulse generated by optical photons
  #endif
  G4int m_nSeptal;				  //!< HDS : record septal penetration

  // AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume
  //These variables no sense for a general pulse but I need them to  process idealy the hits. or create another structure
  G4String m_Postprocess;         // PostStep process
  G4double m_energyIniTrack;         // Initial energy of the track
  G4double m_energyFin;         // final energy of the particle
  G4String m_processCreator;
  G4int m_trackID;
  G4int m_parentID;


  G4double m_energyError;          	  //!< energy error
  G4ThreeVector m_globalPosError;      //!<
  G4ThreeVector m_localPosError;   	  //!<


  G4double m_sourceEnergy;
  G4int m_sourcePDG;
  G4int m_nCrystalConv;
  //--------------------


  //! Pointer to the original crystal hit if known
  const void* m_mother;
};



typedef G4TDigiCollection<GateDigi> GateDigiCollection;

extern G4Allocator<GateDigi> GateDigiAllocator;



inline void* GateDigi::operator new(size_t)
{
  void* aDigi;
  aDigi = (void*) GateDigiAllocator.MallocSingle();
  return aDigi;
}





inline void GateDigi::operator delete(void* aDigi)
{
  GateDigiAllocator.FreeSingle((GateDigi*) aDigi);
}

#endif

