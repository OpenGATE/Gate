

#ifndef GateCCCoincidenceDigi_h
#define GateCCCoincidenceDigi_h 1

#include "G4VDigi.hh"
#include "G4TDigiCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include <fstream>


#include "GatePulse.hh"


class GateCCCoincidenceDigi : public G4VDigi
{

public:

  GateCCCoincidenceDigi();
  GateCCCoincidenceDigi(GatePulse* pulse, G4int coincidenceID);
  GateCCCoincidenceDigi(const GatePulse& pulse, G4int coincidenceID);
  virtual inline ~GateCCCoincidenceDigi() {}

  inline void* operator new(size_t);
  inline void  operator delete(void*);

  void Draw();
  void Print();

  //
  //printing methods
  //
  //friend std::ostream& operator<<(std::ostream&, const GateCCCoincidenceDigi& );

  //friend std::ofstream& operator<<(std::ofstream&, GateCCCoincidenceDigi* );

public:

      inline void  SetRunID(G4int j)                  	      { m_pulse.SetRunID(j); }
      inline G4int GetRunID() const                        	      { return m_pulse.GetRunID(); }

      inline void  SetEventID(G4int j)                	      { m_pulse.SetEventID(j); }
      inline G4int GetEventID() const                      	      { return m_pulse.GetEventID(); }


      inline void     SetTime(G4double value)         	      { m_pulse.SetTime(value); }
      inline G4double GetTime() const                      	      { return m_pulse.GetTime(); }

      inline void SetEnergy(G4double value)           	      { m_pulse.SetEnergy(value); }
      inline G4double GetEnergy() const                    	      { return m_pulse.GetEnergy(); }

      inline void  SetLocalPos(const G4ThreeVector& xyz)      { m_pulse.SetLocalPos(xyz); }
      inline const G4ThreeVector& GetLocalPos() const              { return m_pulse.GetLocalPos(); }

      inline void  SetGlobalPos(const G4ThreeVector& xyz)     { m_pulse.SetGlobalPos(xyz); }
      inline const G4ThreeVector& GetGlobalPos() const             { return m_pulse.GetGlobalPos(); }


      inline void  SetSourcePosition(const G4ThreeVector& xyz)	{ m_pulse.SetSourcePosition(xyz); }
      inline const G4ThreeVector& GetSourcePosition() const          { return m_pulse.GetSourcePosition(); }
      inline void SetSourceEnergy(G4double value)           	      { m_pulse.SetSourceEnergy(value); }
      inline G4double GetSourceEnergy() const                    	      { return m_pulse.GetSourceEnergy(); }
      inline void SetSourcePDG(G4int value)           	      { m_pulse.SetSourcePDG(value); }
      inline G4int GetSourcePDG() const                    	      { return m_pulse.GetSourcePDG(); }
      inline void SetNCrystalConv(G4int value)           	      { m_pulse.SetNCrystalConv(value); }
      inline G4int GetNCrystalConv() const                    	      { return m_pulse.GetNCrystalConv(); }
    // Compton y Rayl
      inline void SetNCrystalCompton(G4int value)           	      { m_pulse.SetNCrystalCompton(value); }
      inline G4int GetNCrystalCompton() const                    	      { return m_pulse.GetNCrystalCompton(); }
      inline void SetNCrystalRayleigh(G4int value)           	      { m_pulse.SetNCrystalRayleigh(value); }
      inline G4int GetNCrystalRayleigh() const                    	      { return m_pulse.GetNCrystalRayleigh(); }
      inline GatePulse& GetPulse()             { return m_pulse; }

       inline void     SetVolumeID(const GateVolumeID& v) {  m_pulse.SetVolumeID(v); }

      inline void     SetCoincidenceID(G4int value)         	      { coincID=value; }
      inline G4double GetCoincidenceID() const                      	      { return coincID; }

      //AE:Ef for AdderComptPhotIdeal Megalib comparison
      inline void     SetFinalEnergy(G4double value)      { m_pulse.SetEnergyFin(value); }
      inline G4double GetFinalEnergy() const                   { return m_pulse.GetEnergyFin(); }

      inline void     SetIniEnergy(G4double value)      { m_pulse.SetEnergyIniTrack(value); }
        inline G4double GetIniEnergy() const                   { return m_pulse.GetEnergyIniTrack(); }



private:
      GatePulse m_pulse;
      G4int coincID;

};





typedef G4TDigiCollection<GateCCCoincidenceDigi> GateCCCoincidenceDigiCollection;

extern G4Allocator<GateCCCoincidenceDigi> GateCCCoincidenceDigiAllocator;





inline void* GateCCCoincidenceDigi::operator new(size_t)
{
  void* aDigi;
  aDigi = (void*) GateCCCoincidenceDigiAllocator.MallocSingle();
  return aDigi;
}





inline void GateCCCoincidenceDigi::operator delete(void* aDigi)
{
  GateCCCoincidenceDigiAllocator.FreeSingle((GateCCCoincidenceDigi*) aDigi);
}

#endif
