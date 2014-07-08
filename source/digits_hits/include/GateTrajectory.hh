/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateTrajectory
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GateTrajectory_h
#define GateTrajectory_h 1

#include "G4VTrajectory.hh"
#include "G4Allocator.hh"
//#include <stdlib.h>
#include "G4ThreeVector.hh"
#include "G4ios.hh"
//#include "g4std"
#include "globals.hh"
#include "G4ParticleDefinition.hh"
#include "G4TrajectoryPoint.hh"
#include "G4Track.hh"
#include "G4Step.hh"

class G4Polyline;

typedef std::vector<G4VTrajectoryPoint*> GateTrajectoryPointContainer;

class GateTrajectory : public G4VTrajectory
{
 public:
   GateTrajectory();
   GateTrajectory(const G4Track* aTrack);
   GateTrajectory(GateTrajectory &);
   virtual ~GateTrajectory();

   inline void* operator new(size_t);
   inline void  operator delete(void*);
   inline int operator == (const GateTrajectory& right) const
   {return (this==&right);}

   inline G4int GetTrackID() const
   { return fTrackID; }
   inline G4int GetParentID() const
   { return fParentID; }
   inline G4String GetParticleName() const
   { return ParticleName; }
   inline G4double GetCharge() const
   { return PDGCharge; }
   inline G4int GetPDGEncoding() const
   { return PDGEncoding; }
   inline const G4ThreeVector& GetMomentum() const
   { return momentum; }
   inline const G4ThreeVector& GetVertexPosition() const
   { return vertexPosition; }
   inline G4double GetGlobalTime() const
   { return globalTime; }
   virtual int GetPointEntries() const
   { return positionRecord->size(); }
   virtual G4VTrajectoryPoint* GetPoint(G4int i) const
   { return (*positionRecord)[i]; }

   //virtual G4ThreeVector GetInitialMomentum() const {return 0;}
   virtual G4ThreeVector GetInitialMomentum() const = 0;

   virtual void ShowTrajectory(std::ostream&) const;
#if (G4VERSION_MAJOR > 9)
   virtual void DrawTrajectory() const;
#else
  virtual void DrawTrajectory(G4int i_mode =0) const;
#endif
   virtual void AppendStep(const G4Step* aStep);
   virtual void MergeTrajectory(G4VTrajectory* secondTrajectory);

   G4ParticleDefinition* GetParticleDefinition();

 private:
   GateTrajectoryPointContainer* positionRecord;
   G4int                        fTrackID;
   G4int                        fParentID;
   G4ParticleDefinition*        fpParticleDefinition;
   G4String                     ParticleName;
   G4double                     PDGCharge;
   G4int                        PDGEncoding;
   G4ThreeVector                momentum;
   G4ThreeVector                vertexPosition;
   G4double                     globalTime;

};

extern G4Allocator<GateTrajectory> myTrajectoryAllocator;

inline void* GateTrajectory::operator new(size_t)
{
  void* aTrajectory;
  aTrajectory = (void*)myTrajectoryAllocator.MallocSingle();
  return aTrajectory;
}

inline void GateTrajectory::operator delete(void* aTrajectory)
{
  myTrajectoryAllocator.FreeSingle((GateTrajectory*)aTrajectory);
}


class GateTrackIDInfo
{
public:
  GateTrackIDInfo(){mParticleName="";mID=0;mParentID=0;}
  GateTrackIDInfo(G4String name, G4int id, G4int pid){mParticleName=name;mID=id;mParentID=pid;}
  ~GateTrackIDInfo(){}

  void SetParticleName(G4String name){mParticleName=name;}
  void SetID(G4int id){mID=id;}
  void SetParentID(G4int id){mParentID=id;}

  G4String GetParticleName(){return mParticleName;}
  G4int GetID(){return mID;}
  G4int GetParentID(){return mParentID;}

protected:
  G4String mParticleName;
  G4int mID;
  G4int mParentID;

};

#endif
