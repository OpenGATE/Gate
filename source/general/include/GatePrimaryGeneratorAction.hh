/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePrimaryGeneratorAction_h
#define GatePrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "GatePrimaryGeneratorMessenger.hh"
#include "globals.hh"

class G4GeneralParticleSource;
class G4Event;
class GatePrimaryGeneratorMessenger;

//---------------------------------------------------------------------------
class GatePrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
  GatePrimaryGeneratorAction();
  ~GatePrimaryGeneratorAction();

public:
  void GeneratePrimaries(G4Event* anEvent);
  void GenerateSimulationPrimaries(G4Event* anEvent);
  void GenerateDigitisationPrimaries(G4Event* anEvent);
  void AddEvent();
  G4int GetEventNumber() { return m_nEvents; };
  //  G4double GetTimeSlice()           { return m_timeSlice; };
  //  void SetTimeSlice(G4double value) { m_timeSlice = value; };
  void SetVerboseLevel(G4int value);
  void EnableGPS(G4bool b) { m_useGPS = b; }

private:
  G4GeneralParticleSource*       m_particleGun;
  G4String                       m_rndmFlag;	    //flag for a random impact point 
  GatePrimaryGeneratorMessenger* m_messenger;
  //    G4double m_timeSlice;

  G4double m_primGenTime;
  G4double m_maxTime;
  G4double m_lifeTime;
  G4int    m_nAliveParticles;
  G4int    m_nEvents;
  G4int    m_nEventsPerRun;
  G4int    m_nTotalEvents;
  G4int    m_printModulo;
  G4int    m_nVerboseLevel;
  G4bool   m_useGPS;
};

#endif



