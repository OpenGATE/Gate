/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details!
  ----------------------*/


#ifndef GateSourceMgr_h
#define GateSourceMgr_h 1

#include "globals.hh"
#include <vector>
#include "G4Event.hh"
#include "G4Run.hh"
#include "GateVSource.hh"

#include "GateApplicationMgr.hh"
#include "GateSourcePencilBeam.hh"
#include "GateSourceTPSPencilBeam.hh"
#include "GateSourceFastY90.hh"

class GateSourceMgrMessenger;

/**
 * @class GateSourceMgr
 *
 * @brief A class to manage multiple sources
 *
 * This class is used to build and use the sources
 * In particular
 *
 *   - it builds the source with AddSource
 *   - it prepare the events for the PrimaryGeneratorAction with PrepareNextEvent
 *   - it initialize the Run variables at PrepareNextRun
 *
 * It has its own internal time, that is updated to the general GATE clock time at
 * the beginning of the Run.
 * For each event, it decides which source is to be used and it asks to this source
 * to generate the primary vertices.
 *
 * GateSourceMgr is a singleton.
 * @author G.Santin
 *
 */

class GateSourceMgr
{
public:
  ~GateSourceMgr();

  // Need to be in header (not in cc)
  static GateSourceMgr* GetInstance() {
    if( mInstance == 0 )
      mInstance = new GateSourceMgr();
    return mInstance;
  }

  G4int AddSource( std::vector<G4String> );
  G4int RemoveSource( G4String name );
  G4int AddSource( GateVSource* pSource );
  GateVSource* GetSourceByName( G4String name );
  GateVSource* GetSource(int i);

  /**
   * @brief to obtain the used sources at the end of the event
   *
   * It permits to ask to the SourceMgr at the end of the event
   * which source(s) has(have) been used for the present
   * event, to set the corresponding flag in the output.
   */
  inline GateVSourceVector GetSourcesForThisEvent()
  { return m_currentSources; }

  /* PY Descourt 08/09/2009 Tracker/Detector */
  G4int GetCurrentSourceID() { return m_currentSourceID; };
  void SetCurrentSourceID( G4int aID ) { m_currentSourceID = aID ; };


  void SetTime( G4double value ) { m_time = value; }
  G4double GetTime() { return m_time; }

  /** It is used internally by PrepareNextEvent
   * to decide which source has to be used for the current event.
   */
  GateVSource* GetNextSource();

  /** It is called by the PrimaryGeneratorAction
   * at each event, to prepare the Primary Vertices.
   */
  G4int PrepareNextEvent( G4Event* event );
	G4bool IsLaunchLastBuffer() { return m_launchLastBuffer; }

  /** It is called by the PrimaryGeneratorAction
   * at the beginning of the Run, to initialize the run-related variables.
   */
  G4int PrepareNextRun( const G4Run* run );

  /** Used by the messenger, command .../source/list */
  void ListSources();

  /** Old method to select a source to change then its attributes.
   * To be eliminated.
   */
  void SelectSourceByName( G4String value );
  void SetVerboseLevel( G4int value );

  void Initialization();

  //void SetIsSuccessiveSources(G4bool t){GateApplicationMgr::GetInstance()->EnableSuccessiveSourceMode(t);}
  //bool IsSuccessiveSourceModeIsEnabled() { return GateApplicationMgr::GetInstance()->IsSuccessiveSourceModeIsEnabled(); }
  bool IsTotalAmountOfPrimariesModeEnabled() { return GateApplicationMgr::GetInstance()->IsTotalAmountOfPrimariesModeEnabled(); }

  //G4double GetTimeSlice(int i)          { return mListOfTimeSlices[i];}
  //G4double GetCurrentTimeSlice()        { return mCurrentSliceTotalTime; }
  //void SetTimeSlice(G4double v);
  //void SetActivity(G4double a);
  //G4double GetActivity(G4int j) {return listOfActivity[j];}
  //void SetNTot(G4int a){m_use_autoweight=true;m_TotNPart=a;}
  void SetWeight(G4double a){mWeight=a;}
  G4double GetWeight(){return mWeight;}

  G4int GetNumberOfEventBySource(int sourceNumber){return mNumberOfEventBySource[sourceNumber];}

  G4int GetSourceID(G4int run ){return mSourceID[run];}
  G4int GetNumberOfSources(){return mSources.size();}

protected:
  GateSourceMgr();
  G4int CheckSourceName( G4String sourceName );

  static GateSourceMgr*     mInstance;
  GateVSourceVector         mSources;
  GateVSource*              m_previousSource;
  GateVSourceVector         m_currentSources;
  GateSourceMgrMessenger*   m_sourceMgrMessenger;
  GateVSource*              m_selectedSource;
  G4bool                    m_needSourceInit;
  G4double                  m_time;
  G4double                  m_timeLimit;
  G4double                  m_timeClock;
	G4double                  m_firstTime;
  G4bool                    m_launchLastBuffer;
	G4int                     m_sourceProgressiveNumber;
  G4int                     mVerboseLevel;
  G4int                     m_TotNPart;
  //  G4int                     m_runNumber;
  G4int                     m_currentSourceNumber;
  //G4bool                    m_use_autoweight;
  //std::vector<G4double>     mListOfTimeSlices;
  //G4double                  mCurrentSliceStartTime;
  //G4double                  mCurrentSliceStopTime;
  //G4double                  mCurrentSliceTotalTime;
  G4int                     mNbOfParticleInTheCurrentRun;
  G4double                  mWeight;
  G4double                  mTotalIntensity;
  //std::vector<G4double>     listOfActivity;
  std::vector<G4int>        listOfWeight;
  std::map<G4int,G4int>     mNumberOfEventBySource;

  std::vector<int>          mSourceID;

  /* PY Descourt 08/09/2008 */
  G4int m_currentSourceID; // for detector mode
  GateVSource* m_fictiveSource; // idem
  G4int p_cK;

};

#endif
