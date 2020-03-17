/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <sstream>

#include "GateConfiguration.h"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4Navigator.hh"
#include "G4UImanager.hh"

#include "Randomize.hh"
#include "GateSourceMgr.hh"
#include "GateSourceMgrMessenger.hh"
#include "GateSourceVoxellized.hh"
#include "GateSourceLinacBeam.hh"
#include "GateClock.hh"
#include "GateApplicationMgr.hh"
#include "GateRTPhantomMgr.hh"
#include <vector>
#include <cmath>
#include "GateActions.hh"
#include "G4RunManager.hh"
#include "GateSourceOfPromptGamma.hh"
#include "GateSourcePhaseSpace.hh"
#include "GateExtendedVSource.hh"

//----------------------------------------------------------------------------------------
GateSourceMgr* GateSourceMgr::mInstance = 0;

//----------------------------------------------------------------------------------------
GateSourceMgr::GateSourceMgr()
{
  mSources.clear();
  m_sourceMgrMessenger = new GateSourceMgrMessenger( this );
  m_selectedSource = 0;
  m_sourceProgressiveNumber = 0;
  mVerboseLevel = 0;
  //m_use_autoweight = false;
//  m_runNumber=-1;
  //mCurrentSliceTotalTime = 0.;
  m_previousSource = 0;
  m_currentSourceNumber=0;
  // PY Descourt 08/09/2009
  m_fictiveSource = new GateVSource("FictiveSource");
  m_currentSourceID = -1;
  mTotalIntensity=0.;
  m_launchLastBuffer = false;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateSourceMgr::~GateSourceMgr()
{
  // loop over the sources to delete them, then clear the pointers vector
  for( size_t i = 0; i != mSources.size(); ++i )//Use an iterator??
    if( mSources[ i ] )
      delete mSources[ i ];
  mSources.clear();
  delete m_sourceMgrMessenger;
  delete m_fictiveSource;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4int GateSourceMgr::AddSource( GateVSource* pSource )
{
  mSources.push_back( pSource );
  return 0;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4int GateSourceMgr::RemoveSource( G4String name )
{
  G4int found = 0;
  if( name == G4String( "all" ) )
    {
      for( size_t is = 0; is != mSources.size(); ++is )//Use an iterator??
        delete mSources[is];
      mSources.clear();
      if( mVerboseLevel > 0 )
        G4cout << "GateSourceMgr::RemoveSource : all sources removed \n";
      return 0;
    }

  GateVSourceVector::iterator itr;
  for( itr = mSources.begin(); itr != mSources.end(); ++itr )
    {
      if( ( *itr )->GetName() == name )
        {
          delete *itr;
          mSources.erase( itr );
          if( mVerboseLevel > 0 )
            G4cout << "GateSourceMgr::RemoveSource : source <" << name
                   << "> removed\n";
          found = 1;
          break;
        }
    }

  if( found == 0 )
    G4cout << "GateSourceMgr::RemoveSource : source <" << name << "> not removed\n";

  return found;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4int GateSourceMgr::AddSource( std::vector<G4String> sourceVec )
{
  if( sourceVec.size() == 0 ) {
    GateError( "GateSourceMgr::AddSource : ERROR : At least the name should be inserted" );
    return 1;
  }

  G4String sourceName     = "";
  G4String sourceGeomType = "";
  //   G4double activity       = 0.;
  //   G4double startTime      = 0.;

  // Loop over the words to download the elements
  G4int nw = sourceVec.size();
  G4bool isGood = true;
  for( G4int iw = 0; ( iw < nw ) && isGood; ++iw ) {
    G4String stranddummy = sourceVec[ iw ] + G4String( " dummy " );
    const char* charWord = stranddummy;
    std::istringstream istrWord( charWord );
    switch(iw) {
    case 0 :
      istrWord >> sourceName;
      if( mVerboseLevel > 0 )
        G4cout << "GateSourceMgr::AddSource : iw " << iw
               << " sourceName " << sourceName << Gateendl;
      break;
    case 1 :
      istrWord >> sourceGeomType;
      if( mVerboseLevel > 0 )
        G4cout << "GateSourceMgr::AddSource : iw " << iw
               << " sourceGeomType " << sourceGeomType << Gateendl;
      break;
    }
    if( mVerboseLevel > 3 )
      G4cout << " istrWord.eof() "  << istrWord.eof()
             << " istrWord.fail() " << istrWord.fail()
             << " istrWord.good() " << istrWord.good()
             << " istrWord.bad() "  << istrWord.bad() << Gateendl;

    if( !istrWord.good() )
      isGood = false;
  }

  G4bool bAddSource = true;
  if (CheckSourceName(sourceName)) bAddSource = false;

  if( bAddSource ) {
    GateVSource* source = 0;
#ifdef G4ANALYSIS_USE_ROOT
    if(sourceGeomType == "phaseSpace"){
      source = new GateSourcePhaseSpace( sourceName );
      source->SetSourceID( m_sourceProgressiveNumber );
      // source->SetActivity( activity );
      //  source->SetStartTime( startTime );
      // dynamic_cast<GateSourcePhaseSpace*>(source)->Initialize();
    }
    else
#endif
      if( sourceGeomType == G4String("voxel") || sourceGeomType == G4String("Voxel") )
        {
          source = new GateSourceVoxellized( sourceName );
          source->SetSourceID( m_sourceProgressiveNumber );
          source->SetIfSourceVoxelized(true);  // added by I. Martinez-Rovira (immamartinez@gmail.com)
        }
      else if ((sourceGeomType == "linacBeam") ||
               (sourceGeomType == "LinacBeam")) {
        source = new GateSourceLinacBeam(sourceName);
        source->SetSourceID( m_sourceProgressiveNumber );
      }
      else if (sourceGeomType == "SourceOfPromptGamma") {
        source = new GateSourceOfPromptGamma(sourceName);
        source->SetSourceID( m_sourceProgressiveNumber );
      }
      else if (sourceGeomType == "PencilBeam") {
        source = new GateSourcePencilBeam( sourceName );
        // source->SetType("PencilBeam");
        source->SetSourceID( m_sourceProgressiveNumber );
      }
      else if (sourceGeomType == "TPSPencilBeam") {
        source = new GateSourceTPSPencilBeam( sourceName );
        // source->SetType("PencilBeam");
        source->SetSourceID( m_sourceProgressiveNumber );
      }
      else if ((sourceGeomType == "gps") ||
               (sourceGeomType == "GPS")) {
        source  =new GateVSource( sourceName );
        source->SetType("gps");
        source->SetSourceID( m_sourceProgressiveNumber );
        source->SetIfSourceVoxelized(false);  // added by I. Martinez-Rovira (immamartinez@gmail.com)
        //mSources.push_back( new GateVSource( sourceName ));
        //mSources[mSources.size()-1]->SetType("gps");
	//mSources[mSources.size()-1]->SetSourceID( m_sourceProgressiveNumber );
      }
      else if (sourceGeomType == "backtoback") {
        source = new GateVSource( sourceName );
        source->SetType("backtoback");
        source->SetSourceID( m_sourceProgressiveNumber );
        source->SetIfSourceVoxelized(false);  // added by I. Martinez-Rovira (immamartinez@gmail.com)
      }
      else if (sourceGeomType == "fastI124") {
        source = new GateVSource( sourceName );
        source->SetType("fastI124");
        source->SetSourceID( m_sourceProgressiveNumber );
      }
      else if (sourceGeomType == "fastY90") {
        source = new GateSourceFastY90( sourceName );
        source->SetType("fastY90");
        source->SetSourceID( m_sourceProgressiveNumber );
      }
      else if (sourceGeomType == "") {
        source = new GateVSource( sourceName );
        source->SetType("gps");
        source->SetSourceID( m_sourceProgressiveNumber );
        source->SetIfSourceVoxelized(false);  // added by I. Martinez-Rovira (immamartinez@gmail.com)
      }
      else if (sourceGeomType == "Extended"){
    	source = new GateExtendedVSource( sourceName );
    	source->SetSourceID( m_sourceProgressiveNumber );
      }
      else {
        GateError("Unknown source type '" << sourceGeomType
                  << "'. Known types are voxel, linacbeam, gps.\n");
      }

    mSources.push_back( source );
    m_sourceProgressiveNumber++;
  }
  else
    G4cout << "GateSourceMgr::AddSource : WARNING: Source not added \n";

  return 0;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4int GateSourceMgr::CheckSourceName( G4String sourceName )
{
  G4int iRet = 0;
  // check if source name already exists
  GateVSourceVector::iterator itr;
  for( itr = mSources.begin(); itr != mSources.end(); ++itr )
    if( ( (*itr)->GetName() ) == sourceName)
      {
        GateError( "GateSourceMgr::CheckSourceName : ERROR : source name <" << sourceName << "> already exists" );
        iRet=1;
      }
  return iRet;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateVSource* GateSourceMgr::GetSourceByName( G4String name )
{
  GateVSource* source = 0;
  GateVSourceVector::iterator itr;
  for( itr = mSources.begin(); itr != mSources.end(); ++itr )
    if( (*itr)->GetName() == name ){
      source = *itr;
      break;
    }

  if( !source )
    GateError( "GateSourceMgr::GetSourceByName : ERROR : source <" << name << "> not found" );

  return source;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateVSource* GateSourceMgr::GetSource(int i) {
  if (i<(int)mSources.size() && i>=0) return mSources[i];
  GateError("The source " << i << " does not exist. Only " << mSources.size() << " are defined.\n");
  return NULL;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateVSource* GateSourceMgr::GetNextSource()
{
  // the method decides which is the source that has to be used for this event
  // static iterator??
  GateVSource* pFirstSource = 0;
  m_firstTime = -1.;

  if( mSources.size() == 0 ) {
    G4cout << "GateSourceMgr::GetNextSource : WARNING: No source available"
           << Gateendl;
    return NULL; // GateError ???
  }

  G4double aTime;

  if (IsTotalAmountOfPrimariesModeEnabled()) {
    G4double randNumber = G4UniformRand()*mTotalIntensity;
    G4double sumIntensity=0.;
    G4int currentSourceNumber = 0;
    while ( (currentSourceNumber<(int)mSources.size()) && (sumIntensity<=randNumber)){
      pFirstSource = mSources[ currentSourceNumber ];
      sumIntensity += pFirstSource->GetIntensity();
      currentSourceNumber++;
    }

    m_firstTime = GateApplicationMgr::GetInstance()->GetTimeStepInTotalAmountOfPrimariesMode();
  }
  else {
    // if there is at least one source
    // make a competition among all the available sources
    // the source that proposes the shortest interval for the next event wins
    GateVSourceVector::iterator itr;
    for( itr = mSources.begin(); itr != mSources.end(); ++itr )
      {
        aTime = (*itr)->GetNextTime( m_time ); // compute random time for this source
        if( mVerboseLevel > 1 )
          G4cout << "GateSourceMgr::GetNextSource : source "
                 << (*itr)->GetName()
                 << "    Next time (s) : " << aTime/s
                 << "   m_firstTime (s) : " << m_firstTime/s << Gateendl;

        if( m_firstTime < 0. || ( aTime < m_firstTime ) )
          {
            m_firstTime = aTime;
            pFirstSource = *itr;
          }
      }
  }

  m_currentSourceID = pFirstSource->GetSourceID(); /* PY Descourt 08/09/2009 */

  return pFirstSource;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceMgr::ListSources()
{
  G4cout << "GateSourceMgr::ListSources: List of the sources in the source manager\n";
  GateVSourceVector::iterator itr;
  for( itr = mSources.begin(); itr != mSources.end(); ++itr )
    ( *itr )->Dump( 0 );
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceMgr::SelectSourceByName( G4String name )
{
  m_selectedSource = GetSourceByName( name );
  if( m_selectedSource ) {
    if( mVerboseLevel > 0 ) {
      G4cout << "GateSourceMgr::SelectSourceByName : source <"
             << name << "> selected\n";
    }
    else {
      G4cout << "GateSourceMgr::SelectSourceByName : WARNING : source <" << name
             << "> not selected\n";
    }
  }
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceMgr::SetVerboseLevel( G4int value ) {
  mVerboseLevel = value;
  GateVSourceVector::iterator itr;
  for( itr = mSources.begin(); itr != mSources.end(); ++itr )
    (*itr)->SetVerboseLevel( mVerboseLevel );
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateSourceMgr::Initialization()//std::vector<G4double> * time, std::vector<G4double> * nparticles)
{
  // GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
  /*  size_t nSliceTot=0;
      for( size_t is = 0; is != mSources.size(); ++is ){
      //mSources[is]->SetActivity();
      int ns = mSources[is]->GetTimePerSlice().size();
      if(ns > 0){
      SetIsSuccessiveSources(true);
      for(int js = 0; js != ns ; ++js ){
      //appMgr->SetTimeInterval(mSources[is]->GetTimePerSlice()[js]);
      //appMgr->SetActivity(mSources[is]->GetNumberOfParticlesPerSlice()[js]/mSources[is]->GetTimePerSlice()[js]);
      mSourceID.push_back(is);
      time->push_back(mSources[is]->GetTimePerSlice()[js]);
      nparticles->push_back(mSources[is]->GetNumberOfParticlesPerSlice()[js]/mSources[is]->GetTimePerSlice()[js]);
      SetActivity(mSources[is]->GetNumberOfParticlesPerSlice()[js]/mSources[is]->GetTimePerSlice()[js]);
      nSliceTot++;
      }
      }
      else if(mListOfTimeSlices.size()>0){
      //appMgr->SetTimeInterval(mListOfTimeSlices[is]);
      //appMgr->SetActivity(listOfActivity[is]);
      time->push_back(mListOfTimeSlices[is]);
      nparticles->push_back(mSources[is]->GetActivity());//listOfActivity[is]);
      SetActivity(mSources[is]->GetActivity());
      mSourceID.push_back(is);
      nSliceTot++;
      }
      }
      G4cout<<"TEST  : Nsources = "<<nSliceTot<< Gateendl;*/

  GateVSourceVector::iterator itr;
  for( itr = mSources.begin(); itr != mSources.end(); ++itr )
    {
      (*itr)->Initialize();
      // mNumberOfEventBySource
      // double intensity =0.;
      if((*itr)->GetIntensity()==0) GateError("Intensity of the source should not be null");
      mTotalIntensity += (*itr)->GetIntensity();// intensity;
    }

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4int GateSourceMgr::PrepareNextRun( const G4Run* r)
{
  //  GateMessage("Acquisition", 0, "PrepareNextRun "  << r->GetRunID() << Gateendl);
  if( mVerboseLevel > 1 )
    G4cout << "GateSourceMgr::PrepareNextRun\n";

  // Initialize the internal time to the GATE clock time
  mNbOfParticleInTheCurrentRun = 0;
  GateClock* theClock = GateClock::GetInstance();
  m_timeClock = theClock->GetTime();
  m_time = m_timeClock;
  m_currentSourceNumber++;
  //G4cout<<"Time Clock = "<<m_time<< Gateendl;
  // Get the next time
  GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
  //G4cout<<"test GetTimeSlice\n";
  G4double timeSlice = appMgr->GetTimeSlice(r->GetRunID());
  //m_timeLimit = m_time + timeSlice;
  m_timeLimit = appMgr->GetEndTimeSlice(r->GetRunID());
  //mCurrentSliceTotalTime = timeSlice;
  //mCurrentSliceStartTime = m_time;
  //mCurrentSliceStopTime = m_timeLimit;
  //   GateMessage("Acquisition", 0,
  //               "Run from t="  << mCurrentSliceStartTime/s << " sec to "
  //               << mCurrentSliceStopTime/s << " sec.\n");

  if( mVerboseLevel > 1 )
    G4cout << "GateSourceMgr::PrepareNextRun : m_time      (s) "
           << m_time/s << Gateendl
           << "                                  m_timeClock (s) "
           << m_timeClock/s << Gateendl
           << "                                  timeSlice   (s) "
           << timeSlice/s << Gateendl
           << "                                  m_timeLimit (s) "
           << m_timeLimit/s << Gateendl;

  //! sending commands to the GateRDM
  G4UImanager* UImanager = G4UImanager::GetUIpointer();
  G4String command;
  UImanager->ApplyCommand( "/grdm/analogueMC 1" );
  UImanager->ApplyCommand( "/grdm/verbose 0" );
  UImanager->ApplyCommand( "/grdm/allVolumes" );

  // set time limit of the GateRDM decay
  command = G4String( "/gate/decay/setDecayTimeLimit " )
    + G4UIcommand::ConvertToString(timeSlice/s) + G4String( " s" );
  if( mVerboseLevel > 3 )
    G4cout << "GateSourceMgr::PrepareNextEvent: command " << command << Gateendl;
  UImanager->ApplyCommand( command.c_str() );
  // tell to the GateRDM to avoid the generation of the sampled decay time for the ions
  // (the time is set by the SourceMgr)
  UImanager->ApplyCommand( "/gate/decay/setPrimaryDecayTimeGeneration 0" );

  // flag for the initialization of the sources
  m_needSourceInit = true;

  // Update the sources (for example for new positioning according to the geometry movements)
  for(GateVSourceVector::iterator itr = mSources.begin(); itr != mSources.end(); ++itr )
    (*itr)->Update(m_time);


//  m_runNumber++;

  // if(m_runNumber==0)
  //     {
  //       DD("TODO !!! autoweight");
  /*
    double totParticles = 0.;
    for( size_t i = 0; i != mSources.size(); ++i ) {
    mSources[i]->SetActivity();
    totParticles += mListOfTimeSlices[i]*listOfActivity[i];
    if(mSources[i]->GetSourceWeight()!=1. && m_use_autoweight)
    GateError("You use the macro 'UseSameNumberOfParticlesPerRun' and you define a manual weight");
    }
    if(m_use_autoweight){
    for( size_t i = 0; i != mSources.size(); ++i ){
    listOfWeight[i]=mListOfTimeSlices[i]*listOfActivity[i]/totParticles;
    mSources[i]->SetSourceWeight(listOfWeight[i]);
    mSources[i]->SetActivity(m_TotNPart/mListOfTimeSlices[i]);
    }
    }
  */
  //  }

  return 0;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4int GateSourceMgr::PrepareNextEvent( G4Event* event )
{
  // GateDebugMessage("Acquisition", 0, "PrepareNextEvent "  << event->GetEventID()
  //                    << " at time " << m_time/s << " sec.\n");

  GateSteppingAction* myAction = (GateSteppingAction *) ( GateRunManager::GetRunManager()->GetUserSteppingAction() );
  TrackingMode theMode =myAction->GetMode();
  m_currentSources.clear();

  G4int numVertices = 0;

  if ( (theMode == TrackingMode::kBoth)  || (theMode == TrackingMode::kTracker) )
    {
      GateRTPhantomMgr::GetInstance()->UpdatePhantoms(m_time); /* PY Descourt 11/12/2008 */

      if( mVerboseLevel > 1 )
        G4cout << "GateSourceMgr::PrepareNextEvent\n";

      // ask the source for this event
      if( mVerboseLevel > 1 )
        G4cout << "GateSourceMgr::PrepareNextEvent : GetNextSource() \n";
      GateVSource* source = GetNextSource();

      if( source )
      {
        // obsolete: to avoid the initialization phase for the source if it's the same as
        // the previous event (always the same with only 1 source). Not needed now with one gps
        // per source
        if( source != m_previousSource ) m_needSourceInit = true;
        m_previousSource = source;

        // save the information, that can then be asked during the analysis phase
        m_currentSources.push_back( source );

        // update the internal time
        m_time += m_firstTime;


        GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
        // G4double timeStop           = appMgr->GetTimeStop();
        appMgr->SetCurrentTime(m_time);

        if( mVerboseLevel > 1 )
          G4cout << "GateSourceMgr::PrepareNextEvent :  m_time (s) " << m_time/s
                 << "  m_timeLimit (s) " << m_timeLimit/s << Gateendl;

        if( m_time <= m_timeLimit || appMgr->IsTotalAmountOfPrimariesModeEnabled() || appMgr->IsReadNumberOfPrimariesInAFileModeEnabled() )
          {
            if( mVerboseLevel > 1 )
              G4cout << "GateSourceMgr::PrepareNextEvent : source selected <"
                     << source->GetName() << ">\n";

            // transmit the time to the source and ask it to generate the primary vertex
            source->SetTime( m_time );
            source->SetNeedInit( m_needSourceInit );
            SetWeight(appMgr->GetWeight());
            source->SetSourceWeight(GetWeight());
            mNumberOfEventBySource[source->GetSourceID()+1]+=1;
            numVertices = source->GeneratePrimaries( event );
          }
        else {
          if( mVerboseLevel > 0 )
            G4cout << "GateSourceMgr::PrepareNextEvent : m_time > m_timeLimit. No vertex generated\n";
        }
      }
      else {
        G4cout << "GateSourceMgr::PrepareNextEvent : WARNING : GateSourceMgr::GetNextSource gave no source\n";
      }

      m_needSourceInit = false;

      mNbOfParticleInTheCurrentRun++;
    } // normal or Tracker Modes

  if ( theMode == TrackingMode::kDetector ) // detector mode
    {
      m_currentSources.push_back(m_fictiveSource);
      //G4cout << "GateSourceMgr::PrepareNextEvent :   m_fictiveSource = " << m_fictiveSource << Gateendl;
      numVertices = m_fictiveSource->GeneratePrimaries(event);
      m_fictiveSource->SetTime(m_time); // time has been set in GeneratePrimaries

      //	G4cout << "GateSourceMgr::PrepareNextEvent :::::::      Time " << m_time/s << " time limit " << m_timeLimit/s << Gateendl;

      if (m_time > m_timeLimit) {  numVertices = 0 ;}

    }

  if( ( m_time + 5.0 * m_firstTime ) > m_timeLimit ) {  m_launchLastBuffer = true;}

  if (mVerboseLevel>1)
    G4cout << "GateSourceMgr::PrepareNextEvent : numVertices : " << numVertices << Gateendl;
  return numVertices;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
/*void GateSourceMgr::SetTimeSlice(G4double time)
  {
  mListOfTimeSlices.push_back(time);
  //GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
  //appMgr->SetTimeInterval(time);
  }*/
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
/*void GateSourceMgr::SetActivity(G4double a)
  {

  listOfActivity.push_back(a);
  //GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
  //appMgr->SetActivity(a);
  }*/
//----------------------------------------------------------------------------------------
