/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateSingleDigi_h
#define GateSingleDigi_h 1

#include "G4VDigi.hh"
#include "G4TDigiCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include <fstream>


#include "GatePulse.hh"


class GateSingleDigi : public G4VDigi
{

public:

  GateSingleDigi();
  GateSingleDigi(GatePulse* pulse);
  GateSingleDigi(const GatePulse& pulse);
  virtual inline ~GateSingleDigi() {}

  inline void* operator new(size_t);
  inline void  operator delete(void*);

  void Draw();
  void Print();

  //
  //printing methods
  //
  friend std::ostream& operator<<(std::ostream&, const GateSingleDigi& );

  friend std::ofstream& operator<<(std::ofstream&, GateSingleDigi* );

public:

      inline void  SetRunID(G4int j)                  	      { m_pulse.SetRunID(j); }
      inline G4int GetRunID() const                        	      { return m_pulse.GetRunID(); }

      inline void  SetEventID(G4int j)                	      { m_pulse.SetEventID(j); }
      inline G4int GetEventID() const                      	      { return m_pulse.GetEventID(); }

      inline void  SetSourceID(G4int j)               	      { m_pulse.SetSourceID(j); }
      inline G4int GetSourceID() const                     	      { return m_pulse.GetSourceID(); }

      inline void  SetSourcePosition(const G4ThreeVector& xyz)	{ m_pulse.SetSourcePosition(xyz); }
      inline const G4ThreeVector& GetSourcePosition() const          { return m_pulse.GetSourcePosition(); }

      inline void     SetTime(G4double value)         	      { m_pulse.SetTime(value); }
      inline G4double GetTime() const                      	      { return m_pulse.GetTime(); }

      inline void SetEnergy(G4double value)           	      { m_pulse.SetEnergy(value); }
      inline G4double GetEnergy() const                    	      { return m_pulse.GetEnergy(); }

      inline void  SetLocalPos(const G4ThreeVector& xyz)      { m_pulse.SetLocalPos(xyz); }
      inline const G4ThreeVector& GetLocalPos() const              { return m_pulse.GetLocalPos(); }

      inline void  SetGlobalPos(const G4ThreeVector& xyz)     { m_pulse.SetGlobalPos(xyz); }
      inline const G4ThreeVector& GetGlobalPos() const             { return m_pulse.GetGlobalPos(); }

      inline void  SetNPhantomCompton(G4int j)        	      { m_pulse.SetNPhantomCompton(j); }
      inline G4int GetNPhantomCompton() const              	      { return m_pulse.GetNPhantomCompton(); }

      inline void  SetNCrystalCompton(G4int j)        	      { m_pulse.SetNCrystalCompton(j); }
      inline G4int GetNCrystalCompton() const              	      { return m_pulse.GetNCrystalCompton(); }

      inline void  SetNPhantomRayleigh(G4int j)        	      { m_pulse.SetNPhantomRayleigh(j); }
      inline G4int GetNPhantomRayleigh() const              	      { return m_pulse.GetNPhantomRayleigh(); }

      inline void  SetNCrystalRayleigh(G4int j)        	      { m_pulse.SetNCrystalRayleigh(j); }
      inline G4int GetNCrystalRayleigh() const              	      { return m_pulse.GetNCrystalRayleigh(); }

      inline void     SetComptonVolumeName(const G4String& name) {  m_pulse.SetComptonVolumeName(name); }
      inline G4String GetComptonVolumeName() const            	 { return m_pulse.GetComptonVolumeName(); }

      inline void     SetRayleighVolumeName(const G4String& name) {  m_pulse.SetRayleighVolumeName(name); }
      inline G4String GetRayleighVolumeName() const            	 { return m_pulse.GetRayleighVolumeName(); }

      inline void  SetScannerPos(const G4ThreeVector& xyz)    { m_pulse.SetScannerPos(xyz); }
      inline const G4ThreeVector& GetScannerPos() const            { return m_pulse.GetScannerPos(); }

      inline void     SetScannerRotAngle(G4double value)      { m_pulse.SetScannerRotAngle(value); }
      inline G4double GetScannerRotAngle() const                   { return m_pulse.GetScannerRotAngle(); }

      inline const GateOutputVolumeID& GetOutputVolumeID() const    { return m_pulse.GetOutputVolumeID(); }
      inline G4int GetComponentID(size_t depth) const    { return m_pulse.GetComponentID(depth); }

      inline void     SetVolumeID(const GateVolumeID& v) {  m_pulse.SetVolumeID(v); }

    

      inline GatePulse& GetPulse()             { return m_pulse; }

      // HDS : septal penetration
      inline void  SetNSeptal(G4int n)    { m_pulse.SetNSeptal(n); }
      inline G4int GetNSeptal() const     { return m_pulse.GetNSeptal(); }

	//AE: to use offline the idealAdderComptPhot and recover the initial energy of the photon (initial energy of the primary track) and the energy after the interaction (Megalib comparison)
      inline void     SetFinalEnergy(G4double value)      { m_pulse.SetEnergyFin(value); }
      inline G4double GetFinalEnergy() const                   { return m_pulse.GetEnergyFin(); }

      inline void     SetIniEnergy(G4double value)      { m_pulse.SetEnergyIniTrack(value); }
      inline G4double GetIniEnergy() const                   { return m_pulse.GetEnergyIniTrack(); }


      inline void SetSourceEnergy(G4double value)           	      { m_pulse.SetSourceEnergy(value); }
      inline G4double GetSourceEnergy() const                    	      { return m_pulse.GetSourceEnergy(); }

      inline void SetSourcePDG(G4int value)           	              { m_pulse.SetSourcePDG(value); }
      inline G4int GetSourcePDG() const                    	      	      { return m_pulse.GetSourcePDG(); }

      inline void SetNCrystalConv(G4int value)           	              { m_pulse.SetNCrystalConv(value); }
      inline G4int GetNCrystalConv() const                    	      	      { return m_pulse.GetNCrystalConv(); }




  
     

private:
      GatePulse m_pulse;

public:
  static void SetSingleASCIIMask(G4bool);
  static void SetSingleASCIIMask(std::vector<G4bool>);
  static std::vector<G4bool> GetSingleASCIIMask();
  static G4bool GetSingleASCIIMask(G4int index);

protected:
  static std::vector<G4bool> m_singleASCIIMask;
  static G4bool                m_singleASCIIMaskDefault;
};





typedef G4TDigiCollection<GateSingleDigi> GateSingleDigiCollection;

extern G4Allocator<GateSingleDigi> GateSingleDigiAllocator;





inline void* GateSingleDigi::operator new(size_t)
{
  void* aDigi;
  aDigi = (void*) GateSingleDigiAllocator.MallocSingle();
  return aDigi;
}





inline void GateSingleDigi::operator delete(void* aDigi)
{
  GateSingleDigiAllocator.FreeSingle((GateSingleDigi*) aDigi);
}

#endif

