/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class  GateRunAction, GateEventAction, GateTrackingAction, GateSteppingAction
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEACTION_HH
#define GATEACTION_HH

#include "G4UserRunAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserStackingAction.hh"

#include "globals.hh"
#include "GateApplicationMgr.hh"
#include "G4TrackStatus.hh"

class GateTrack;
class GateVVolume;
class GateUserActions;

enum class TrackingMode {
  kUnknown, kBoth,  kTracker,
  kDetector
};

//-----------------------------------------------------------------------------
/// \brief G4UserRunAction which redirects its callbacks to GateUserActions
class GateRunAction :  public G4UserRunAction
{
public :
  GateRunAction(GateUserActions * cbm);
  ~GateRunAction(){}

  //-----------------------------------------------------------------------------
  // Action classes Callbacks

  void BeginOfRunAction(const G4Run* aRun);
  inline void EndOfRunAction(const G4Run* aRun);

  inline virtual void SetFlagBasicOutput( G4bool val ) { flagBasicOutput = val;};
  inline G4bool GetFlagBasicOutput () { return flagBasicOutput; };

  virtual inline void SetRunAction ( GateRunAction* val ) { prunAction = val; };
  static inline GateRunAction* GetRunAction() { return prunAction; };
  //-----------------------------------------------------------------------------

private:
  GateRunAction() {}
  GateUserActions* pCallbackMan;

  G4int runIDcounter;
  G4bool flagBasicOutput;
  static GateRunAction* prunAction;

  //OK GND 2022
public: // also used by GateEventAction
     std::vector<G4int> m_CHCollIDs; //vector of HitCollectionID from the case of multiple SDs
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// \brief G4UserEventAction which redirects its callbacks to GateUserActions
class GateEventAction :  public G4UserEventAction
{
public :
  GateEventAction(GateUserActions * cbm);
  ~GateEventAction()
  {}
  //-----------------------------------------------------------------------------
  // Action classes Callbacks

  void BeginOfEventAction(const G4Event* anEvent);
  void EndOfEventAction(const G4Event* anEvent);

  virtual inline void SetFlagBasicOutput( G4bool val ) { flagBasicOutput = val; };
  virtual inline G4bool GetFlagBasicOutput () { return flagBasicOutput; };

  virtual inline void SetEventAction ( GateEventAction* val ) { peventAction = val; };
  static inline GateEventAction* GetEventAction() { return peventAction; };
  //-----------------------------------------------------------------------------
private:
  GateEventAction() {}
  GateUserActions* pCallbackMan;

  G4bool flagBasicOutput;
  static GateEventAction* peventAction;
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// \brief G4UserTrackingAction which redirects its callbacks to GateUserActions
class GateTrackingAction :  public G4UserTrackingAction
{
public :
  GateTrackingAction(GateUserActions * cbm);
  //    : pCallbackMan(cbm){}
  ~GateTrackingAction() {}
  //-----------------------------------------------------------------------------
  // Action classes Callbacks
  void PreUserTrackingAction(const G4Track* a) ;
  void PostUserTrackingAction(const G4Track* a);
  void ShowG4TrackInfos( G4String,G4Track* );
  //-----------------------------------------------------------------------------
private:
  GateTrackingAction() {}
  GateUserActions* pCallbackMan;

  /* PY Descourt 08/09/2009 */
  std::vector<G4Track*> dummy_track_vector;
  std::vector<G4Step*> dummy_step_vector;
  /* PY Descourt 08/09/2009 */
};
//-----------------------------------------------------------------------------
class GateSteppingActionMessenger;
//-----------------------------------------------------------------------------
/// \brief G4UserSteppingAction which redirects its callbacks to GateUserActions
class GateSteppingAction :  public G4UserSteppingAction
{
public :
  GateSteppingAction(GateUserActions * cbm);
  ~GateSteppingAction() ;
  //-----------------------------------------------------------------------------
  // Action classes Callbacks
  //Modif Seb 24/02/2009 // Remis par Tibo et David (02-06-2009), non mais !
  void UserSteppingAction(const G4Step* a);
  void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void SetDrawTrajectoryLevel(G4int value) { m_drawTrjLevel = value; };
  virtual void SetVerboseLevel(G4int value)        { m_verboseLevel = value; };

  inline  std::vector<GateTrack*> *GetPPTrackVector() { return PPTrackVector;};

  void StopOnBoundary(G4int aI);
  void StopAndKill(G4String aString);
  void SetMode( TrackingMode aMode);
  TrackingMode GetMode();
  void SetTxtOut(G4String aString);
  G4int GetTxtOn() { return TxtOn;};
  void SetEnergyThreshold(G4double);
  void SetFiles(G4int aN ) { m_Nfiles = aN; };
  G4int GetNfiles() { return m_Nfiles;};
  G4int GetcurrentN() { return m_currentN; };
  G4int SeekNewFile( G4bool increase );
  void SetRSFiles(G4int aN ) { m_NfilesRS = aN; };
  G4int GetNRSfiles() { return m_NfilesRS;};
  G4int GetcurrentNRS() { return m_currentNRS; };
  G4int SeekNewRecStepFile( G4bool increase );
  G4int NoMoreRecStepData() { return ( m_currentNRS == m_NfilesRS - 1 ); };
  G4int NoMoreTracksData()  { return ( m_currentN == m_Nfiles - 1 ); };
  void ShowG4TrackInfos( G4String  outF, G4Track* aTrack);

  //-----------------------------------------------------------------------------
private:
  GateSteppingAction() {}
  GateUserActions* pCallbackMan;
  G4int m_drawTrjLevel;
  G4int m_verboseLevel;

protected :

  ////// PY Descourt 08/09/2009 //
  //
  GateSteppingActionMessenger* m_steppingMessenger;
  std::vector<GateTrack*> *PPTrackVector;
  TrackingMode m_trackingMode;
  G4int Boundary; // if set to 1 stop track on Phantom Boundary
  G4int fKeepOnlyP;
  G4int fKeepOnlyPhotons;
  G4int fKeepOnlyElectrons;
  G4TrackStatus fStpAKill;
  G4int TxtOn;
  G4int m_Nfiles; // in detector mode sets the number of Tracks Root Files
  G4int m_currentN; // current file opened
  G4int m_NfilesRS; // in detector mode sets the number of phantom hits  Root Files
  G4int m_currentNRS; // current phantom hits file opened
  G4String m_StartingVolName; // In TRACKER MODE ONLY : this is the volume where the particle is created
  G4bool fKillNextIsSet;                      //  In TRACKER MODE ONLY : a flag to be sure to kill the particle JUST after the boundary ( Boundary == 0 case )
  G4bool fStartVolumeIsPhantomSD; // is the outgoing particle was created in a Phantom Type Sensitive Detector ?
  G4double m_energyThreshold;
  //
  //////
};
//-----------------------------------------------------------------------------


#endif
