/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceDigi_h
#define GateCoincidenceDigi_h 1

#include "G4VDigi.hh"
#include "G4TDigiCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"


#include "GateCoincidencePulse.hh"


class GateCoincidenceDigi : public G4VDigi
{

public:

  GateCoincidenceDigi();
  GateCoincidenceDigi(GateCoincidencePulse* coincidencePulse);
  GateCoincidenceDigi(const GateCoincidencePulse& coincidencePulse);
  virtual inline ~GateCoincidenceDigi() {}

  inline void* operator new(size_t);
  inline void  operator delete(void*);

  void Draw();
  void Print();

  //
  //printing methods
  //
  friend std::ostream& operator<<(std::ostream&, GateCoincidenceDigi&);

  friend std::ofstream& operator<<(std::ofstream&, GateCoincidenceDigi*);

public:

      inline GatePulse& GetPulse(G4int i)                     { return pulseVector[i]; }
      inline void SetPulse(G4int i, const GatePulse& value)   { pulseVector[i] = value; }

private:

      GatePulse pulseVector[2];

public:
  static void SetCoincidenceASCIIMask(G4bool);
  static void SetCoincidenceASCIIMask(std::vector<G4bool>);
  static std::vector<G4bool> GetCoincidenceASCIIMask();
  static G4bool GetCoincidenceASCIIMask(G4int index);

protected:
  static std::vector<G4bool> m_coincidenceASCIIMask;
  static G4bool                m_coincidenceASCIIMaskDefault;
};





typedef G4TDigiCollection<GateCoincidenceDigi> GateCoincidenceDigiCollection;

extern G4Allocator<GateCoincidenceDigi> GateCoincidenceDigiAllocator;





inline void* GateCoincidenceDigi::operator new(size_t)
{
  void* aDigi;
  aDigi = (void*) GateCoincidenceDigiAllocator.MallocSingle();
  return aDigi;
}





inline void GateCoincidenceDigi::operator delete(void* aDigi)
{
  GateCoincidenceDigiAllocator.FreeSingle((GateCoincidenceDigi*) aDigi);
}

#endif
