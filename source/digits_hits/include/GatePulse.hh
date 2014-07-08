/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulse_h
#define GatePulse_h 1

#include "GateConfiguration.h"
#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"

/*! \class  GatePulse
    \brief  Class for storing a 'pulse' (luminous or electronic) derived from one or more hits

    - GatePulse - by Daniel.Strul@iphe.unil.ch

    - When hits are registered, theyr are converted into pulses by a GateHitConvertor

    - These pulses may then be processed by a series of pulse-processors. The output of this
      processing is a pulse-list, which is then converted into digis.

    - S. Stute: june2014, add two methods used in the new GateReadout implementation

      \sa GateVPulseProcessor, GatePulseProcessorChain
*/
class GateVSystem;

class GatePulse
{
  public:
    //! Constructor
    GatePulse(const void* itsMother=0);

    //! Copy constructor
    inline GatePulse(const GatePulse* right)
    {
      	*this = *right;
    }
    //! Destructor
    virtual inline ~GatePulse() {}

  public:
    //! \name getters and setters to acces the content of the pulse
    //@{

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

      //! S. Stute: Modify one value inside the VolumeID and outputVolumeID vectors
      void ChangeVolumeIDAndOutputVolumeIDValue(size_t depth, G4int value);
      // Reset the local position to be 0
      inline void ResetLocalPos() {m_localPos[0]=0.;m_localPos[1]=0.;m_localPos[2]=0.;}
      void ResetGlobalPos(GateVSystem* system);


#ifdef GATE_USE_OPTICAL
      inline void   SetOptical(G4bool optical = true) { m_optical = optical;}
      inline G4bool IsOptical() const { return m_optical;}
#endif

	// HDS : record septal penetration
	  inline G4int GetNSeptal() const { return m_nSeptal; }
	  inline void SetNSeptal(G4int septalNb) { m_nSeptal = septalNb; }
    //@}

      virtual const GatePulse& CentroidMerge(const GatePulse* secondaryPulse);
      virtual const GatePulse& CentroidMergeCompton(const GatePulse* right);
    //! printing methods
    friend std::ostream& operator<<(std::ostream&, const GatePulse&);

private:
  //! \name pulse data
  //@{
  G4int m_runID;      	      	  //!< runID
  G4int m_eventID;            	  //!< eventID
  G4int m_sourceID;           	  //!< source progressive number
  G4ThreeVector m_sourcePosition; //!< position of the source (NOT the positron) that generated the hit
  G4double m_time;            	  //!< start time of the current pulse
  G4double m_energy;          	  //!< energy measured for the current pulse
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
#ifdef GATE_USE_OPTICAL
  G4bool m_optical;               //!< Is the pulse generated by optical photons
#endif
	G4int m_nSeptal;				  //!< HDS : record septal penetration
  //@}

  //! Pointer to the original crystal hit if known
  const void* m_mother;

};


/*! \class  GatePulseList
    \brief  List of pulses

    - GatePulseList - by Daniel.Strul@iphe.unil.ch

    - These lists are generated and processed by pulse-processors

      \sa GatePulse, GateHitConvertor, GateVPulseProcessor
*/
class GatePulseList: public std::vector<GatePulse*>
{
  public:
    inline GatePulseList(const G4String& aName)
      : m_name(aName)
      {}
    GatePulseList(const GatePulseList& src);
    virtual ~GatePulseList();

    //! Return the min-time of all pulses
    virtual GatePulse* FindFirstPulse() const ;
    virtual G4double ComputeStartTime() const ;
    virtual G4double ComputeFinishTime() const ;
    G4double ComputeEnergy() const;
    virtual void InsertUniqueSortedCopy(GatePulse* newPulse);

    //! Return the list-name
    const G4String& GetListName() const
      {return m_name;}
    void SetName(const G4String& name){m_name=name;}

  protected:
    G4String m_name;
};


//! Iterator on a pulse list
typedef GatePulseList::iterator GatePulseIterator;


//! Constant iterator on a pulse list
typedef GatePulseList::const_iterator GatePulseConstIterator;


#endif
