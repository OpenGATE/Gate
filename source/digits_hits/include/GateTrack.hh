#ifndef GateTrack_H
#define GateTrack_H

#include "G4ThreeVector.hh"
#include "G4ios.hh"
#include "globals.hh"
#include "G4Track.hh"
#include <vector>

#include "TROOT.h"
#include "TTree.h"


//#include "GatePhantomHit.hh"

class GateTrack
{  public:

   GateTrack();
   GateTrack( GateTrack& aTrack);

   ~GateTrack();

   G4bool Compare( G4Track* aTrack);

   G4int GetTrackID();
   void SetTrackID(const G4int aValue);

   G4int GetEventID();
   void SetEventID(const G4int aValue);

   G4int GetRunID();
   void SetRunID(const G4int aValue);

void SetProcessName(G4String aName ) { m_processName = aName; };
G4String GetProcessName() { return m_processName; };

  G4String GetParentParticleName(){ return m_ParentParticleName;};
  void SetParentParticleName(  G4String aName ){m_ParentParticleName = aName;};

  G4String GetParticleName ();
  void SetParticleName(  G4String aName ) ;
   G4int GetParentID();
   void SetParentID(const G4int aValue);

   // position, time
   G4ThreeVector& GetPosition();
   void SetPosition(const G4ThreeVector& aValue);

   G4double GetGlobalTime();
   void SetGlobalTime(const G4double aValue);
     // Time since the event in which the track belongs is created.

   G4double GetLocalTime();
   void SetLocalTime(const G4double aValue);
      // Time since the current track is created.

   G4double GetProperTime();
   void SetProperTime(const G4double aValue);
      // Proper time of the current track
   G4double GetKineticEnergy();
   void SetKineticEnergy(const G4double aValue);

   G4double GetTotalEnergy();
   void SetTotalEnergy(G4double Energy);

  // moemtnum
    G4ThreeVector& GetMomentumDirection();
   void SetMomentumDirection(const G4ThreeVector& aValue);

   G4ThreeVector GetMomentum();
   void SetMomentum(G4ThreeVector aVector);

   G4double GetVelocity();
   void SetVelocity(G4double aVelocity);

  // polarization
   G4ThreeVector& GetPolarization() ;
   void SetPolarization(const G4ThreeVector& aValue);

  // vertex (,where this track was created) information
    G4ThreeVector& GetVertexPosition() ;
   void SetVertexPosition(G4ThreeVector& aValue);

    G4ThreeVector& GetVertexMomentumDirection() ;
   void SetVertexMomentumDirection(const G4ThreeVector& aValue);

   G4double GetVertexKineticEnergy() ;
   void SetVertexKineticEnergy(const G4double aValue);

   G4int GetPDGCode(){ return m_PDGCode;};
   void SetPDGCode( G4int aCode ) { m_PDGCode = aCode; };


  // track weight
  // These are methods for manipulating a weight for this track.
   G4double GetWeight() ;
   void     SetWeight(G4double aValue);

   void SetVertexVolumeName( G4String aName);
   G4String GetVertexVolumeName();

//   std::vector<GatePhantomHit> GetPHitVector(){return PHitVector;};

void Print();
void Fill_Track(G4Track* aTrack);
G4int GetSourceID() {return m_sourceID;};
void SetSourceID(G4int aID) { m_sourceID = aID; };
void SetWasKilled( G4int aI) { fwasKilled = aI; };
G4int GetWasKilled() { return fwasKilled;};
G4double GetTime() { return m_time; };
void SetTime (G4double aTime) { m_time = aTime; };

private:

   G4double m_time;  // time at which the tarck is created
   G4ThreeVector fPosition;        // Current positon
   G4double fGlobalTime;           // Time since the event is created
   G4double fLocalTime;            // Time since the track is created
   G4double fProperTime;
   G4int fParentID;
   G4int feventID;
   G4int fRunID;
   G4int fTrackID;
   G4double fKineticEnergy;
   G4ThreeVector fMomentum;
   G4ThreeVector fMDirection;
   G4double fVelocity;
  G4ThreeVector fPolarization;
   G4double fTotalEnergy;
   G4ThreeVector fVtxPosition;          // (x,y,z) of the vertex
   G4ThreeVector fVtxMomentumDirection; // Momentum direction at the vertex
   G4double fVtxKineticEnergy;          // Kinetic energy at the vertex
   G4double fWeight;
   G4String ParticleName;
   G4String VertexVolumeName;
   G4int fwasKilled;  //////////////// is 1 if the particle was killed in the phantom
                 ////////////////  needed because it may happen that a primary particle
                ////////////////   does not go out of the phantom so we are in trouble when erasing GateTracks from TrackVector

   G4String m_processName,m_ParentParticleName;

   G4int m_PDGCode,m_sourceID;

//   std::vector<GatePhantomHit> PHitVector;

};

#endif
