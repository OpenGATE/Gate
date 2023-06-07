/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCoincidenceDigi_h
#define GateCoincidenceDigi_h 1

#include "G4VDigi.hh"
#include "G4TDigiCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include <fstream>
#include <iterator>

#include "GateDigi.hh"


// define the minimum offset for a delayed coincidence window in sec
#define  MIN_COINC_OFFSET -1.

class GateCoincidenceDigi : public G4VDigi, public std::vector<GateDigi*>
{
public:

	GateCoincidenceDigi(const void* itsMother=0);

	GateCoincidenceDigi(GateDigi *firstDigi,
						G4double itsCoincidenceWindow,
						G4double itsOffsetWindow);
	//GateCoincidenceDigi(GateCoincidencePulse* coincidencePulse);
	GateCoincidenceDigi(const GateCoincidenceDigi& src);
	inline ~GateCoincidenceDigi() {};

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


    inline G4double GetStartTime() const
      { return m_startTime; }

    inline G4double GetEndTime() const
      { return m_endTime; }
/*
   inline G4int GetCoincID() const
      { return m_coincID; }
   inline void SetCoincID(int coincID) 
      { m_coincID=coincID; }

*/

   virtual G4bool IsInCoincidence(const GateDigi* newDigi) const;
   virtual G4bool IsAfterWindow(const GateDigi* newDigi) const;

   inline G4bool IsDelayed() const
         { return m_delayed;}


  private:
    G4double m_startTime;
    G4double m_endTime;
    G4bool m_delayed;
    G4int m_coincID;
public:
   GateDigi* GetDigi(G4int i);
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
