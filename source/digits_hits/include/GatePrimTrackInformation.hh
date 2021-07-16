

#ifndef GatePrimTrackInformation_h
#define GatePrimTrackInformation_h 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4Track.hh"
#include "G4Allocator.hh"
#include "G4VUserTrackInformation.hh"

class GatePrimTrackInformation : public G4VUserTrackInformation 
{
public:
  GatePrimTrackInformation();
  GatePrimTrackInformation(const G4Track* aTrack);
  GatePrimTrackInformation(const GatePrimTrackInformation* aTrackInfo);
  virtual ~GatePrimTrackInformation();
   
  inline void *operator new(size_t);
  inline void operator delete(void *aTrackInfo);

 GatePrimTrackInformation& operator =(const GatePrimTrackInformation& right);
  
  void SetEPrimTrackInformation(const G4Track* aTrack);
  virtual void Print() const;

public:
  
  inline G4double GetSourceEini() const {return m_energyPrimaryTrack;}
  inline int GetSourcePDG() const {return m_PDGPrimaryTrack;}


   inline void setNCompton(const G4int nCompt) {m_NCompton=nCompt;}
   inline int GetNCompton() const {return m_NCompton;}
  
   inline void setNConv(const G4int numConv) {m_NConv=numConv;}
   inline int GetNConv() const {return m_NConv;}

   inline void setNRayl(const G4int numRayl) {m_NRayl=numRayl;}
   inline int GetNRayl() const {return m_NRayl;}

private:
  // Information of the primary track at the primary vertex
  G4int                 fOriginalTrackID;  // Track ID of primary particle
  G4ParticleDefinition* fParticleDefinition;
  G4ThreeVector         fOriginalPosition;
  G4ThreeVector         fOriginalMomentum;
  G4double              fOriginalEnergy;
  G4double              fOriginalTime;

  G4double             m_energyPrimaryTrack;
  G4int                m_PDGPrimaryTrack;

  G4int                m_NCompton;
  G4int                m_NConv;
  G4int                m_NRayl;





};

extern G4ThreadLocal
 G4Allocator<GatePrimTrackInformation> * aTrackInformationAllocator;

inline void* GatePrimTrackInformation::operator new(size_t)
{
  if(!aTrackInformationAllocator)
    aTrackInformationAllocator = new G4Allocator<GatePrimTrackInformation>;
  return (void*)aTrackInformationAllocator->MallocSingle();
}

inline void GatePrimTrackInformation::operator delete(void *aTrackInfo)
{ aTrackInformationAllocator->FreeSingle((GatePrimTrackInformation*)aTrackInfo);}

#endif

