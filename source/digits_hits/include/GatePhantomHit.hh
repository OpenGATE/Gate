/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GatePhantomHit_h
#define GatePhantomHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "GateConfiguration.h"

class GatePhantomHit : public G4VHit
{
  public:

      GatePhantomHit();
      ~GatePhantomHit();

      inline void *operator new(size_t);
      inline void operator delete(void *aHit);

      void Draw();
      void Print();

private:
  G4int m_PDGEncoding;   // G4 PDGEncoding
  G4double m_edep;       // energy deposit for the current hit
  G4double m_stepLength; // length of the step for the current hit
  G4double m_time;       // time of the current hit
  G4ThreeVector m_pos;   // position of the current hit
  G4String m_process;    // process on the current hit
  G4int m_trackID;       // track ID
  G4int m_parentID;      // parent track ID

  G4int  m_voxelCoordinates;  //  voxellized phantom voxel number
  G4String m_physVolName;

// v. cuplov - optical photons
//  static const G4String theOutputAlias;
// v. cuplov - optical photons

  public:
      inline void SetEdep(G4double de)          { m_edep = de; }
      inline void AddEdep(G4double de)          { m_edep += de; }

      inline G4double GetEdep()               { return m_edep; }

      inline void SetStepLength(G4double value) { m_stepLength = value; }
      inline G4double GetStepLength()           { return m_stepLength; }

      inline void     SetTime(G4double aTime)    { m_time = aTime; }
      inline G4double GetTime()                 { return m_time; }

      inline void          SetPos(G4ThreeVector xyz)     { m_pos = xyz; }
      inline G4ThreeVector GetPos()                     { return m_pos; }

      inline void     SetProcess(G4String proc) { m_process = proc; }
      inline G4String GetProcess()              { return m_process; }

      inline void  SetPDGEncoding(G4int j)      { m_PDGEncoding = j; }
      inline G4int GetPDGEncoding()            { return m_PDGEncoding; }

      inline void  SetTrackID(G4int j)          { m_trackID = j; }
      inline G4int GetTrackID()                 { return m_trackID; }

      inline void  SetParentID(G4int j)         { m_parentID = j; }
      inline G4int GetParentID()                { return m_parentID; }

      inline void SetVoxelCoordinates(G4int c)  { m_voxelCoordinates = c ;   }
      inline G4int  GetVoxelCoordinates()const  { return m_voxelCoordinates; }

      inline void SetPhysVolName(G4String name) { m_physVolName = name ;   }
      inline G4String GetPhysVolName()const     { return m_physVolName; }

// v. cuplov - optical photons
      inline G4bool GoodForAnalysis() const
      	  { return ( (m_process != "Transportation") || (m_edep!=0.) ); }

//     static  const G4String& GetOutputAlias() {return theOutputAlias;}
// v. cuplov - optical photons

};

typedef G4THitsCollection<GatePhantomHit> GatePhantomHitsCollection;

extern G4Allocator<GatePhantomHit> GatePhantomHitAllocator;

inline void* GatePhantomHit::operator new(size_t)
{
  void *aHit;
  aHit = (void *) GatePhantomHitAllocator.MallocSingle();
  return aHit;
}

inline void GatePhantomHit::operator delete(void *aHit)
{
  GatePhantomHitAllocator.FreeSingle((GatePhantomHit*) aHit);
}

#endif
