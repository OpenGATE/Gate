/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4Gamma.hh"
#include "G4GenericIon.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"

#include "GateBackToBack.hh"
#include "GateFastI124.hh"
#include "GateClock.hh"
#include "GateMessageManager.hh"
#include "Randomize.hh"
#include "GateObjectStore.hh"
#include "GateSourceMgr.hh"
#include "GateMiscFunctions.hh"
#include "GateActions.hh"
#include "GateToRoot.hh"
#include "GateOutputMgr.hh"

#include "GateVisManager.hh"
#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4VisAttributes.hh"
// Setup a static color table for source visualization

//-------------------------------------------------------------------------------------------------
// Setup a static color table for source visualization
#define N_COLORCODES 10
GateVSource::GateColorPair GateVSource::theColorTable[N_COLORCODES] = {
    GateColorPair ("white",      G4Colour(1.0, 1.0, 1.0)),
    GateColorPair ("gray",       G4Colour(0.5, 0.5, 0.5)),
    GateColorPair ("grey",       G4Colour(0.5, 0.5, 0.5)),
    GateColorPair ("black",      G4Colour(0.0, 0.0, 0.0)),
    GateColorPair ("red",        G4Colour(1.0, 0.0, 0.0)),
    GateColorPair ("green",      G4Colour(0.0, 1.0, 0.0)),
    GateColorPair ("blue",       G4Colour(0.0, 0.0, 1.0)),
    GateColorPair ("cyan",       G4Colour(0.0, 1.0, 1.0)),
    GateColorPair ("magenta",    G4Colour(1.0, 0.0, 1.0)),
    GateColorPair ("yellow",     G4Colour(1.0, 1.0, 0.0))
  };
GateVSource::GateColorMap GateVSource::theColorMap =
      GateColorMap(N_COLORCODES,theColorTable);

//-------------------------------------------------------------------------------------------------
GateVSource::GateVSource(G4String name): m_name( name ) {
  m_type        			 = "";
  m_sourceID     			 = 0;
  m_activity     			 = 0.*becquerel;
  m_startTime    			 = 0.*s;
  m_time         			 = 0.*s;
  m_timeInterval = 0.*s;
  nVerboseLevel  			 = 0;
  m_NbOfParticles = 0;
  m_weight = 1.;
  mVolume = 0;
  m_intensity = 1;

  m_accolinearityFlag = false;
  m_accoValue = 0.;

  m_forcedUnstableFlag  = false;
  m_forcedLifeTime      = -1.*s;
  m_materialName = "Air";
  mRelativePlacementVolumeName = "world";
  mEnableRegularActivity = false;

  mSourceTime = 0.*s;

  mIsUserFluenceActive = false;
  mUserFluenceFilename = "";
  mCentreCoords = G4ThreeVector(0., 0., 0.);
  mRotX = CLHEP::HepXHat;
  mRotY = CLHEP::HepYHat;
  mRotZ = CLHEP::HepZHat;
//   mUserPosRndm = NULL;
  mIsUserFocalShapeActive = false;
  mUserFocalShapeInitialisation = false;
  mUserFocalShape = new G4SPSPosDistribution();
  mUserPosGenX = 0;

  m_posSPS = new GateSPSPosDistribution();
  m_posSPS->SetBiasRndm( GetBiasRndm() );
  m_eneSPS = new GateSPSEneDistribution();
  m_eneSPS->SetBiasRndm( GetBiasRndm() );
  m_angSPS = new GateSPSAngDistribution();
  m_angSPS->SetPosDistribution( m_posSPS );
  m_angSPS->SetBiasRndm( GetBiasRndm() );

  m_sourceMessenger = new GateVSourceMessenger( this );
  m_SPSMessenger    = new GateSingleParticleSourceMessenger( this );

  SetNumberOfParticles(1); // important !

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateVSource::~GateVSource()
{
  delete m_sourceMessenger;
  delete m_SPSMessenger;

  delete m_posSPS;
  delete m_eneSPS;
  delete m_angSPS;

  delete mUserFocalShape;
  if(mUserPosGenX != 0)
    delete mUserPosGenX;
  for (unsigned int i = 0; i<mUserPosGenY.size(); ++i)
  {
    if(mUserPosGenY[i] != 0)
      delete mUserPosGenY[i];
  }

}
//-------------------------------------------------------------------------------------------------

#ifndef G4VIS_USE
void GateVSource::Visualize(G4String){
#endif
#ifdef G4VIS_USE
void GateVSource::Visualize(G4String parmString){
  G4Tokenizer parms(parmString);
  G4String    sCount=parms();
  G4String    sColor=parms();
  G4String    sSize=parms();
  int   iCount  = atoi(sCount);
  float fSize   = atof(sSize);
 //G4SingleParticleSource *m_sps;

  G4VVisManager *visman = GateVisManager::GetConcreteInstance();
  if (!visman) return;

  if (iCount <=0 || iCount > 10000) {
     G4cout << "Invalid count; 2000 used.\n";
  }

  if(fSize <=0 || fSize > 20) {
    G4cout << "Invalid size; 3.0 used.\n";
  }

  GateColorMap::iterator colorMapIt = theColorMap.find(sColor);
  if ( colorMapIt == theColorMap.end()) {
    G4cout << "Color name '" << sColor << "' was not recognised, yellow used instead.\n";
    colorMapIt = theColorMap.find("yellow");
  }

  G4VisAttributes attribs(colorMapIt->second);

  G4Circle circle;
  circle.SetScreenSize(fSize);
  circle.SetFillStyle(G4Circle::filled);
  circle.SetVisAttributes(attribs);

  for (int k=0; k<iCount; ++k){

    //m_sps->GeneratePositionStuff();

	//Fix to update source visualization when it is attached to a volume
	//M Chamberland, 20/09/2013
	G4ThreeVector position = m_posSPS->GenerateOne();
	ChangeParticlePositionRelativeToAttachedVolume(position);
	circle.SetPosition(position);

	visman->Draw(circle);
  }

#endif
}
//-------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateVSource::EnableRegularActivity(bool b) {
  mEnableRegularActivity = b;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateVSource::SetTimeActivityFilename(G4String filename) {
  ReadTimeDoubleValue(filename, "Activity", mTimeList, mActivityList);
  //DD(mTimeList.size());
  if (mTimeList.size()<1) {
    GateError("While readin time-activity file 'filename', no time ?\n");
  }
  m_activity = mTimeList[0];
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
//void GateVSource::AddTimeSlicesFromFile(G4String /*filename //WARNING: parameter not used */) {
/* //ReadTimeDoubleValue(filename, "Activity", mTimeList, mActivityList);

//mTimePerSlice;
//mNumberOfParticlesPerSlice;

}*/
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
/*void GateVSource::AddTimeSlices(double time, int nParticles) {
//ReadTimeDoubleValue(filename, "Activity", mTimeList, mActivityList);

mTimePerSlice.push_back( time);
mNumberOfParticlesPerSlice.push_back( nParticles);

}*/
//----------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4String GateVSource::GetRelativePlacementVolume() {
  return mRelativePlacementVolumeName;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSource::SetRelativePlacementVolume(G4String volname) {
  mRelativePlacementVolumeName = volname;
  // Search for volume
  mVolume =   GateObjectStore::GetInstance()->FindVolumeCreator(volname);
//mVolume->Describe();
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
//G4double GateVSource::GetNextTimeInSuccessiveSourceMode(G4double /*timeStart*/,
/*                                                        G4int mNbOfParticleInTheCurrentRun) {
                                                          if (mNbOfParticleInTheCurrentRun == 0) {
                                                          return (1.0/m_activity)/2.0;
                                                          }
                                                          return (1.0/m_activity);
                                                          }*/
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
G4double GateVSource::GetNextTime( G4double timeStart )
{

/* GetVolumeID ??? */



  // returns the proposed time for the next event of this source, sampled from the
  // source time distribution
  G4double aTime = DBL_MAX;

//if(m_activity==0 && m_timeInterval!=0.)  SetActivity();

  if( m_activity > 0. )
    {
      // compute the present activity, on the base of the starting activity and the lifetime (if any)
      G4double activityNow = m_activity;
      if( timeStart < m_startTime )
        activityNow = 0.;
      else
        {
	  // Force life time to 0, time is managed by GATE not G4
	  GetParticleDefinition()->SetPDGLifeTime(0);
	  if( m_forcedUnstableFlag )
            {
	      if( m_forcedLifeTime > 0. )
	       {
                activityNow = m_activity *
                  exp( - ( timeStart - m_startTime ) / m_forcedLifeTime );
	       }
	      else
	      {
                G4cout << "[GateVSource::GetNextTime] ERROR: Forced decay with negative lifetime: (s) "
                       << m_forcedLifeTime/s << Gateendl;
	      }
            }
          else
            {
              G4ParticleDefinition* partDef = GetParticleDefinition();
              if( partDef )
                {
                  if( !( partDef->GetPDGStable() ) )
                    {
                      if( nVerboseLevel > 0 )
		      G4cout << "GateVSource::GetNextTime : unstable particle "
			  << GetParticleDefinition()->GetParticleName()
			  << " from source " <<  GetName() << Gateendl;
		      // activity is constant
		      activityNow = m_activity;
                    }
                  else
                    if( nVerboseLevel > 1 )
                      G4cout << "GateVSource::GetNextTime : stable particle "
                             << GetParticleDefinition()->GetParticleName()
                             << " from source " <<  GetName() << Gateendl;
                }
              else
                if (nVerboseLevel>0)
                  G4cout << "GateVSource::GetNextTime : NULL ParticleDefinition for source "
                         << GetName() << " assumed stable \n";
            }
        }
      if( nVerboseLevel > 0 )
        G4cout << "GateVSource::GetNextTime : Initial activity (becq) : "
               << m_activity/becquerel << Gateendl
               << "                            At time (s) " << timeStart/s
               << " activity (becq) " << activityNow/becquerel << Gateendl;

      // sampling of the interval distribution
      if (!mEnableRegularActivity) {
        aTime = -log( G4UniformRand() ) * ( 1. / activityNow );
      }
      else {
        GateError("I should not be here. ");
        // DD(activityNow);
        //         DD(m_activity);
        //         DD(timeStart/s);
        aTime = 1./activityNow;
      }
    }

  if( nVerboseLevel > 0 )
    G4cout << "GateVSource::GetNextTime : next time (s) " << aTime/s << Gateendl;


//Dump(0);
/*G4cout<< "    CentreCoords       (mm)  : "
             << m_posSPS->GetCentreCoords().x()/mm << " "
             << m_posSPS->GetCentreCoords().y()/mm << " "
             << m_posSPS->GetCentreCoords().z()/mm << Gateendl;*/
  return aTime;
}
//-------------------------------------------------------------------------------------------------

void GateVSource::TrigMat()
{
// Retrieve position according to world
  GateVVolume * v = mVolume;
  G4cout<<"------------------|||||||||||| TEST de SEBES =          "<<v->GetObjectName()<< Gateendl;

}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateVSource::Dump( G4int level )
{

 G4cout << "Source --------------> " << m_name << Gateendl
         << "  ID                 : " << m_sourceID << Gateendl
         << "  type               : " << m_type << Gateendl
         << "  activity (Bq)      : " << m_activity/becquerel << Gateendl
         << "  startTime (s)      : " << m_startTime/s << Gateendl
         << "  time (s)           : " << m_time/s << Gateendl
         << "  forcedUnstable     : " << m_forcedUnstableFlag << Gateendl
         << "  forcedHalfLife (s) : " << GetForcedHalfLife()/s << Gateendl
         << "  verboseLevel       : " << nVerboseLevel << Gateendl
         << "  relative to vol    : " << mRelativePlacementVolumeName << Gateendl
         << "---------------------- \n"
         << Gateendl;
  if( level > 0 )
    {
      G4cout << "    GPS info ----------------> \n";
      if( GetParticleDefinition() )
        G4cout << "    particle                 : "
               << GetParticleDefinition()->GetParticleName()
               << Gateendl ;
      else
        G4cout << "    particle                 : "
               << "not defined"
               << Gateendl ;

      G4cout << "    SourcePosType            : "
             << m_posSPS->GetPosDisType() << Gateendl
             << "    Shape                    : "
             << m_posSPS->GetPosDisShape() << Gateendl
             << "    halfx,halfy,halfz  (mm)  : "
             << m_posSPS->GetHalfX()/mm << " "
             << m_posSPS->GetHalfY()/mm << " "
             << m_posSPS->GetHalfZ()/mm << Gateendl
             << "    Radius             (mm)  : "
             << m_posSPS->GetRadius()/mm << Gateendl
             << "    CentreCoords       (mm)  : "
             << m_posSPS->GetCentreCoords().x()/mm << " "
             << m_posSPS->GetCentreCoords().y()/mm << " "
             << m_posSPS->GetCentreCoords().z()/mm << Gateendl
             << "    EnergyDisType            : "
             << m_eneSPS->GetEnergyDisType() << Gateendl
             << "    AngleDisType            : "
             << m_angSPS->GetDistType() << Gateendl
             << "    MinTheta, MaxTheta (deg) : "
             << m_angSPS->GetMinTheta()/deg << " " << m_angSPS->GetMaxTheta()/deg
             << Gateendl
             << "    MinPhi, MaxPhi     (deg) : "
             << m_angSPS->GetMinPhi()/deg << " " << m_angSPS->GetMaxPhi()/deg
             << Gateendl
             << "    -------------------------- \n"
             << Gateendl;
    }
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSource::GeneratePrimariesForBackToBackSource(G4Event* event) {
  // Gammas Pair with GPS
  GateBackToBack* backToBack = new GateBackToBack( this );
  backToBack->Initialize();
  backToBack->GenerateVertex( event, m_accolinearityFlag);
  if( nVerboseLevel > 1 )
    G4cout << "GetNumberOfPrimaryVertex : "
           << event->GetNumberOfPrimaryVertex() << Gateendl;
  if( nVerboseLevel > 1 )
    G4cout << "GetNumberOfParticle      : "
           << event->GetPrimaryVertex(0)->GetNumberOfParticle() << Gateendl;

  delete backToBack;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSource::GeneratePrimariesForFastI124Source(G4Event* event) {
  // Fast I124 : generates 0 to 3 particles (gammas and e+) according to a simplified decay scheme
  // No atomic deexcitation occurs
  GateFastI124* fastI124 = new GateFastI124( this );

  if (!( fastI124->GetSimplifiedDecay())) fastI124->InitializeFastI124();

  fastI124->GenerateVertex(event);

  delete fastI124;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4int GateVSource::GeneratePrimaries( G4Event* event )
{
  if (event) GateMessage("Beam", 2, "Generating particle " << event->GetEventID() << Gateendl);

  G4int numVertices = 0;

  GateSteppingAction* myAction = (GateSteppingAction *) ( GateRunManager::GetRunManager()->GetUserSteppingAction() );

  TrackingMode theMode =myAction->GetMode();

  G4bool test = (theMode ==1 ) || ( theMode == 2 );
  if ( test == 1  )
    {
      if (GetType() == G4String("backtoback"))    { GeneratePrimariesForBackToBackSource(event); }
      else if (GetType() == G4String("fastI124")) { GeneratePrimariesForFastI124Source(event); }
      else if ((GetType() == G4String("")) || (GetType() == G4String("gps"))) {
        // decay time for ions inside the timeSlice controlled here and not by RDM
        // NB: temporary: secondary ions of the decay chain not properly treated
        SetParticleTime( m_time );
        GeneratePrimaryVertex( event );

      }
      else {
        GateError("Sorry, I don't know the source type '"<< GetType() << "'. Known source types are"
                  << "<backtoback> <fastI124> <gps>");
      }
      numVertices++;

      if (event) {
        for(int i=0; i< event->GetPrimaryVertex(0)->GetNumberOfParticle(); i++) {
          G4PrimaryParticle  * p = event->GetPrimaryVertex(0)->GetPrimary(i);
          GateMessage("Beam", 3, "(" << event->GetEventID() << ") " << p->GetG4code()->GetParticleName()
                      << " pos=" << event->GetPrimaryVertex(0)->GetPosition()
                      << " weight=" << p->GetWeight()
                      << " energy=" <<  G4BestUnit(mEnergy, "Energy")
                      << " mom=" << p->GetMomentum()
                      << " ptime=" <<  G4BestUnit(p->GetProperTime(), "Time")
                      << " atime=" <<  G4BestUnit(GetTime(), "Time")
                      << ")\n");
        }
      }

      //if (event) {
      //  printf("time %e ns\n", GetTime());
      //}


      //G4cout<<"Generate primaries\n";
      return numVertices;
    }// standard or tracker mode PY Descourt 08/09/2008

  if ( theMode == 3 )// detector mode     here we have a fictive source
    {
      if ( fAbortNow == true ) { fAbortNow= false;
        return 0; }


      std::vector<GateTrack*>* aTrackVector = myAction->GetPPTrackVector();

      //G4cout << "  GateSource::GeneratePrimaries   OK DEtector Mode   size of Tracks Vector " << aTrackVector->size()<< Gateendl;

      // Check the current Tracks Vector is empty ! otherwise something went wrong /////////
      if ( aTrackVector->size() > 0 )
        {
          std::vector<GateTrack*>::iterator iter;
          for ( iter = aTrackVector->begin(); iter != aTrackVector->end() ; iter++){(*iter)->Print();}
          G4Exception( "GateSource::GeneratePrimaries", "GeneratePrimaries", FatalException, "ERROR : The tracks Vector is not empty.\n");
        }

      /// READ DATA FROM ROOT FILE
      GateToRoot* gateToRoot = (GateToRoot* ) ( GateOutputMgr::GetInstance()->GetModule("root") );
      if ( gateToRoot == 0 )
        { G4cout <<" GateSource::GeneratePrimaries ERROR : In DETECTOR MODE : NO GateToRoot Module...Cannot retrieve Tracker Data \n";
          exit(1);
        }
      if ( gateToRoot->CheckEOF() == 1 ) // end of File reached
        {
          // check if there is some more Tracks Root File to Read
          G4cout << " GateSource::GeneratePrimaries   End of Tracks Root file reached ... Seeking for more files to open.\n";
          G4int test_next = myAction->SeekNewFile(true);
          if ( test_next == 0 )
            {
              numVertices = 0;
              return numVertices;
            }
        }
      // Read the Root File current track
      //
      GateTrack* m_currentTrack = gateToRoot->GetCurrentTracksData();
      if ( m_currentTrack == 0 ) // means we have no more data in the ROOT file so we are done for the current file
        {
          gateToRoot->CloseTracksRootFile();
          numVertices = 0;
          return numVertices;
        }
      G4Run* currentRun = const_cast<G4Run*> ( GateRunManager::GetRunManager()->GetCurrentRun() );
      currentRun->SetRunID( m_currentTrack->GetRunID() );
      event->SetEventID( m_currentTrack->GetEventID() );
      G4int event_id =  m_currentTrack->GetEventID();
      G4int eventID;
      G4int RunID;
      G4ThreeVector DirectionMomentum;
      G4ThreeVector Momentum;
      G4double Weight;
      G4double properTime;
      G4double TotalEnergy;
      G4ThreeVector Polarization;
      G4ThreeVector Position;
      G4int TrackID;
      G4int ParentID;
      G4String ParticleName;
      //G4int sourceID;
      G4int test = 1;
      G4bool id1 = true;
      while ( test == 1 )
        {
          if ( m_currentTrack == 0 )test = 0;
          else
            {
              id1 =  ( m_currentTrack->GetRunID() ==  currentRun->GetRunID() );
              G4bool id2 =  ( m_currentTrack->GetEventID() ==  event_id ) ;
              if (  ( id1 == true )  && ( id2 == true ) )
                {
                  GateTrack*  TmpTrack = new GateTrack( *m_currentTrack );
                  aTrackVector->push_back( TmpTrack );
                  // Read next buffer of Data from Root File
                  gateToRoot->ReadForward();
                  m_currentTrack = gateToRoot->GetCurrentTracksData();
                }
              else test = 0;
            }
        }
      if ( id1 == false ) { fAbortNow = true; }
      std::vector<GateTrack*>::iterator iter = aTrackVector->begin();
      size_t k = 0;
      G4int m_previous_SourceID = (*iter)->GetSourceID();
      for ( iter = aTrackVector->begin(); iter != aTrackVector->end() ; iter++)
        {
          G4int PDGCode = (*iter)->GetPDGCode();
          DirectionMomentum = (*iter)->GetMomentumDirection() ;
          eventID = (*iter)->GetEventID();
          if ( eventID != event_id ){G4cout << " GateSource::GeneratePrimaries()   GateTrack # "<<k<<" event_ID is "<<eventID<<"   current event_ID is " << event_id<< Gateendl;G4Exception( "GateVSource::GeneratePrimaries","GeneratePrimaries",FatalException,"ABORTING...");}
          RunID = (*iter)->GetRunID();
          Weight = (*iter)->GetWeight();
          properTime = (*iter)->GetProperTime();
          TotalEnergy = (*iter)->GetTotalEnergy();
          Polarization = (*iter)->GetPolarization() ;
          Position =  (*iter)->GetPosition();
          TrackID = (*iter)->GetTrackID();
          ParentID = (*iter)->GetParentID();
          Momentum = (*iter)->GetMomentum();
          /// Source Infos
         // sourceID = (*iter)->GetSourceID();
          fPosition = (*iter)->GetVertexPosition();
          m_sourceID = (*iter)->GetSourceID(); // we set the source ID to the current one
          if ( m_sourceID != m_previous_SourceID )
            { G4cout << "GateSource::GeneratePrimaries :::: ERROR "<< Gateendl;
              G4cout << "GateSource::GeneratePrimaries :::: Run ID " << RunID << " Event ID " << eventID <<" source ID " << m_sourceID << " event Time " <<  (GateSourceMgr::GetInstance())->GetTime()/s << "  track ID " << TrackID << "   parent ID " << ParentID <<    Gateendl;
              if ( k == aTrackVector->size() ){G4Exception( "GateSource::GeneratePrimaries", "GeneratePrimaries", FatalException, "The sources ID of primaries do not correspond");}
            }

          G4double eventTime = (*iter)->GetTime();
          (GateSourceMgr::GetInstance())->SetTime( eventTime );
          SetTime(eventTime);
          /// we generate 1 particle for each  PrimaryVertex we got from Tracks Root file
          G4ParticleTable  *particleTable = G4ParticleTable::GetParticleTable();
          m_pd = particleTable->FindParticle( PDGCode );
          if( m_pd == NULL){G4Exception( "Gate::GeneratePrimaries", "GeneratePrimaries", FatalException, "ERROR PDGCode of the particle  is not defined. \n"); }
          ParticleName = (G4String) ( m_pd->GetParticleName() );
          (*iter)->SetParticleName( ParticleName );
          SetNumberOfParticles(1);
          GeneratePrimaryVertex(event);
          /// Get the  particle from kth PrimaryVertex which is the one just generated
          event->GetPrimaryVertex(k)->SetPosition(Position.x() , Position.y() ,Position.z() );
          G4PrimaryParticle* particle = event->GetPrimaryVertex(k)->GetPrimary(0);
          /// Sets its kinetics and dynamical variables
          particle->Set4Momentum(DirectionMomentum.x(),DirectionMomentum.y(),DirectionMomentum.z(), TotalEnergy );
          particle->SetWeight( Weight) ;
          particle->SetProperTime(properTime);
          particle->SetPolarization(Polarization.x(),Polarization.y(),Polarization.z());
          particle->SetTrackID(TrackID);
          particle->SetMomentum(Momentum.x(),Momentum.y(),Momentum.z());
          k++;
          numVertices++;
          m_previous_SourceID = m_sourceID;
        }
    } //                 Detector Mode - PY Descourt 08/09/2009

  return numVertices;

}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateVSource::Update(double t)
{
  m_time = t;
  if( nVerboseLevel > 0 )
    G4cout << "[GateVSource::Update] Source name: " << m_name << Gateendl;
  // called by the sourceMgr at the beginning of the run.
  // when a mechanism for the attachment of a source to a geometry volume will be in place
  // if the source is "attached" to a volume here it should update its own position according
  // to the (new) position of the volume.

  // if the activity change according to time, set it
  if (mTimeList.size() != 0) {
    //DD(m_time/s);
    int i = GetIndexFromTime(mTimeList, m_time);
    //DD(i);
    m_activity = mActivityList[i];
  }
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateVSource::GeneratePrimaryVertex( G4Event* aEvent )
{
  if( GetParticleDefinition() == NULL ) return;
  if( GetPosDist()->GetPosDisType() == "UserFluenceImage" ) InitializeUserFluence();
  if( mUserFocalShapeInitialisation ) InitializeUserFocalShape();

  if( nVerboseLevel > 1 ) {
    G4cout << " NumberOfParticlesToBeGenerated: " << GetNumberOfParticles() << Gateendl ;
  }

  /* PY Descourt 08/09/2009 */
  TrackingMode theMode =( (GateSteppingAction *)(GateRunManager::GetRunManager()->GetUserSteppingAction() ) )->GetMode();
  if (  theMode == kBoth || theMode == kTracker )
    {
      G4ThreeVector particle_position;
      if(mIsUserFluenceActive) { particle_position = UserFluencePosGenerateOne(); }
      else { particle_position = m_posSPS->GenerateOne(); }

      // Set placement relative to attached volume
      ChangeParticlePositionRelativeToAttachedVolume(particle_position);

      G4PrimaryVertex* vertex = new G4PrimaryVertex(particle_position, GetParticleTime());
      if (GetNumberOfParticles() == 0) {
        GateError("Something went wrong, nb of particle is 0 in GateVSource::GeneratePrimaryVertex\n");
      }

      for( G4int i = 0 ; i != GetNumberOfParticles() ; ++i )
        {
          G4ParticleMomentum particle_momentum_direction;
          if(mIsUserFocalShapeActive) { particle_momentum_direction = UserFocalShapeGenerateOne(); }
          else { particle_momentum_direction = m_angSPS->GenerateOne(); }

          // Set placement relative to attached volume
          ChangeParticleMomentumRelativeToAttachedVolume(particle_momentum_direction);
          // DD(particle_momentum_direction);

          G4double particle_energy = 0;
          particle_energy = m_eneSPS->GenerateOne( GetParticleDefinition() );
          mEnergy = particle_energy; // because particle_energy is private

          G4double mass =  GetParticleDefinition()->GetPDGMass();
          G4double energy = particle_energy + mass;
          G4double pmom = std::sqrt( energy * energy - mass * mass );
          G4double px = pmom * particle_momentum_direction.x();
          G4double py = pmom * particle_momentum_direction.y();
          G4double pz = pmom * particle_momentum_direction.z();

          G4PrimaryParticle* particle = new G4PrimaryParticle(GetParticleDefinition(), px, py, pz);
          particle->SetMass( mass );
          particle->SetCharge( GetParticleDefinition()->GetPDGCharge() );
          particle->SetPolarization( GetParticlePolarization().x(),
                                     GetParticlePolarization().y(),
                                     GetParticlePolarization().z() );

          G4double particle_weight = GetBiasRndm()->GetBiasWeight();
          particle->SetWeight( particle_weight );

          // Add one particle
          vertex->SetPrimary( particle );

	  // Verbose
          if( nVerboseLevel > 1 ) {
            G4cout << "Particle name: " << GetParticleDefinition()->GetParticleName() << Gateendl;
            G4cout << "       Energy: " << particle_energy << Gateendl ;
            G4cout << "     Position: " << particle_position << Gateendl ;
            G4cout << "    Direction: " << particle_momentum_direction << Gateendl;
          }
          if( nVerboseLevel > 2 ) {
            G4cout << "Creating primaries and assigning to vertex\n";
          }
        } // end loop on NumberOfParticles

/*G4StepPoint point1;
G4ThreeVector position(particle_position.x(),particle_position.y(),particle_position.z());
point1.SetPosition(position);
G4cout << particle_position.x()/mm << " " << particle_position.y()/mm << " " << particle_position.z()/mm << Gateendl;
G4cout << point1.GetPosition().x()/mm << " " << point1.GetPosition().y()/mm << " " << point1.GetPosition().z()/mm << Gateendl;
G4Material* material = point1.GetMaterial();
G4String nameMaterial = material->GetName();*/
//G4cout<<"####### Material Name du step : "<<point1.GetMaterial()->GetName()<< Gateendl;

      aEvent->AddPrimaryVertex( vertex );
    }

  /////  HERE we are in DETECTOR MODE
  /* PY Descourt 08/09/2009 */

  if ( theMode == kDetector )
    {
      // create a new vertex
      G4PrimaryVertex* vertex =  new G4PrimaryVertex(G4ThreeVector(0.,0.,0.),GetTime());

      if(GetVerboseLevel() > 0)
        G4cout << "Creating primaries and assigning to vertex\n";

      for( G4int i=0; i<GetNumberOfParticles(); i++ )
        {
          G4PrimaryParticle* particle = new G4PrimaryParticle(m_pd,0. ,0. , 0.);
          vertex->SetPrimary( particle );
        }

      aEvent->AddPrimaryVertex( vertex );
    }
  if( nVerboseLevel > 1 )
    G4cout << " Primary Vertex generated !\n";
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateVSource::ChangeParticlePositionRelativeToAttachedVolume(G4ThreeVector & position) {
  // Do nothing if attached to world
  if (mRelativePlacementVolumeName == "world") return;

  // Current position
  GateMessage("Beam", 4, "Current particle position = " << position << Gateendl);

  // Retrieve position according to world
  GateVVolume * v = mVolume;
  while (v->GetObjectName() != "world") {
    G4RotationMatrix r = v->GetPhysicalVolume(0)->GetObjectRotationValue();
    const G4ThreeVector & t = v->GetPhysicalVolume(0)->GetObjectTranslation();
    position = r*position;
    position = position+t;
    GateMessage("Beam", 4, "Change current particle position = " << position << Gateendl);
    // next volume
    v = v->GetParentVolume();
  }
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateVSource::ChangeParticleMomentumRelativeToAttachedVolume(G4ParticleMomentum & momentum) {

  // Do nothing if attached to world
  if (mRelativePlacementVolumeName == "world") return;

  // Current position
  GateMessage("Beam", 4, "Current particle mom = " << momentum << Gateendl);

  // Retrieve rotation according to world
  GateVVolume * v = mVolume;
  while (v->GetObjectName() != "world") {
    // DD(v->GetObjectName());
    // DD(v->GetPhysicalVolume(0)->GetObjectTranslation());
    // DD(v->GetPhysicalVolume(0)->GetObjectRotationValue());
    G4RotationMatrix r = v->GetPhysicalVolume(0)->GetObjectRotationValue();
    //const G4ThreeVector & t = v->GetPhysicalVolume(0)->GetObjectTranslation();
    momentum = r*momentum;
    // next volume
    v = v->GetParentVolume();
  }
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
  void GateVSource::SetIonDefaultHalfLife()
{
    // This method is to set the default half life to the used ion source when we use "useDefaultHalfLife" command.
  if(m_SPSMessenger->GetIonShooting() == 1)
   {
      if(m_forcedUnstableFlag == true)
         m_forcedLifeTime = GetParticleDefinition()->GetPDGLifeTime();
   }
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/*void GateVSource::SetTimeInterval(double time)
  {
  DD("SetTimeInterval(t)");
  mSourceTime = time;
  GateSourceMgr::GetInstance()->SetIsSuccessiveSources(true);
  GateSourceMgr::GetInstance()->SetTimeSlice(time);
  m_timeInterval = time;
  }*/
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
/*void GateVSource::SetActivity(){
  DD("SetActivity()");

  if(m_timeInterval==0) m_timeInterval = GateSourceMgr::GetInstance()->GetCurrentTimeSlice();
  if(m_timeInterval==0) GateError("The time interval is null!");
  if(m_activity==0.) m_activity = m_NbOfParticles/m_timeInterval;
  GateSourceMgr::GetInstance()->SetActivity(m_activity);
  }*/
//-------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateVSource::InitializeUserFluence()
{
  GateMessage("Beam", 0, "WARNING message :\n");
  GateMessage("Beam", 0, "You are using a userFluenceImage source. Type, shape, size are automatically extracted from the image.\n");

  if(mUserFluenceFilename == "")
  {
    GateError("You are using a userFluenceImage source. Please, enter a filename to be read (../gps/pos/setImage)");
  }
  else
  {
    mIsUserFluenceActive = true;

    // Define the 'official' randomPosEngine as a point type source (known in G4SPSPosDistribution)
    // position is randomly selected using an alternative randomEngine according to the userFluenceImage
    // WARNING : units in userImage have to be in mm
    m_posSPS->SetPosDisType("Point");

    GateImage userFluenceImage;
    userFluenceImage.Read(mUserFluenceFilename);

    int resX = userFluenceImage.GetResolution().x();
    int resY = userFluenceImage.GetResolution().y();
    double sizeX = userFluenceImage.GetVoxelSize().x() * mm;
    double sizeY = userFluenceImage.GetVoxelSize().y() * mm;
    mUserFluenceVoxelSize.set(userFluenceImage.GetVoxelSize().x(),
			      userFluenceImage.GetVoxelSize().y(),
			      userFluenceImage.GetVoxelSize().z());

    mUserPosX.resize(resX);
    mUserPosY.resize(resY);
    mUserPosGenY.resize(resX);
    double sum;

    // Generate XBias and "YBias knowing X" according to fluence image
    double posX,posY;
    posX = (0.5 * sizeX) + userFluenceImage.GetOrigin().x();
//     posX = ((0.5 * sizeX) - userFluenceImage.GetHalfSize().x());

    mUserPosGenX = new G4SPSRandomGenerator();

    mUserPosGenX->SetXBias(G4ThreeVector(0.,0.,0.));
    for(int i=0; i<resX;i++)
    {
      mUserPosX[i] = posX;

      sum = 0.0;
      posY = (0.5 * sizeY) + userFluenceImage.GetOrigin().y();
//       posY = ((0.5 * sizeY) - userFluenceImage.GetHalfSize().y());

      if (mUserPosGenY[i]==0) {
        mUserPosGenY[i] = new G4SPSRandomGenerator();
        mUserPosGenY[i]->SetYBias(G4ThreeVector(0.,0.,0.));
        mUserPosGenY[i]->GetBiasWeight();       // these two lines are kludges to initialize
        mUserPosGenY[i]->ReSetHist("biaspp");   // all the G4Cache objects in G4SPSRandomGenerator
      }

      for(int j=0; j<resY; j++)
      {
        sum += userFluenceImage.GetValue(i,j,0);
        mUserPosY[j] = posY;

        mUserPosGenY[i]->SetYBias(G4ThreeVector(j+1,userFluenceImage.GetValue(i,j,0),0.));
        posY += sizeY;
      }

      mUserPosGenX->SetXBias(G4ThreeVector(i+1,sum,0.));
      posX += sizeX;
    }

  }
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
G4ThreeVector GateVSource::UserFluencePosGenerateOne()
{
  int i = floor(mUserPosGenX->GenRandX());
  int j = floor(mUserPosGenY[i]->GenRandY());

  // uniform rand in pixel
  double x = mUserPosX[i] + (G4UniformRand()-0.5)*mUserFluenceVoxelSize.x();
  double y = mUserPosY[j] + (G4UniformRand()-0.5)*mUserFluenceVoxelSize.y();
  double z = 0.;

  // This is a partial copy of G4SPSPosDistribution::GeneratePointsInPlane(...)
  // - Apply Rotation Matrix
  // - x * Rotx, y * Roty and z * Rotz
  double tempx = (x * mRotX.x()) + (y * mRotY.x()) + (z * mRotZ.x());
  double tempy = (x * mRotX.y()) + (y * mRotY.y()) + (z * mRotZ.y());
  double tempz = (x * mRotX.z()) + (y * mRotY.z()) + (z * mRotZ.z());

  G4ThreeVector randPos;
  randPos.setX(tempx);
  randPos.setY(tempy);
  randPos.setZ(tempz);

  // - Apply translation
  randPos += mCentreCoords;

  // Moving the source (point) and call GenerateOne function
  m_posSPS->SetCentreCoords(randPos);
  G4ThreeVector position = m_posSPS->GenerateOne();

  return position;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateVSource::SetCentreCoords(G4ThreeVector coordsOfCentre)
{
  mCentreCoords = coordsOfCentre;
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateVSource::SetPosRot1(G4ThreeVector posrot1)
{
  mRotX = posrot1;

  // This is a copy of G4SPSPosDistribution::GenerateRotationMatrices()
  mRotX = mRotX.unit(); // x'
  mRotY = mRotY.unit(); // vector in x'y' plane
  mRotZ = mRotX.cross(mRotY); // z'
  mRotZ = mRotZ.unit();
  mRotY = mRotZ.cross(mRotX); // y'
  mRotY = mRotY.unit();
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateVSource::SetPosRot2(G4ThreeVector posrot2)
{
  mRotY = posrot2;

  // This is a copy of G4SPSPosDistribution::GenerateRotationMatrices()
  mRotX = mRotX.unit(); // x'
  mRotY = mRotY.unit(); // vector in x'y' plane
  mRotZ = mRotX.cross(mRotY); // z'
  mRotZ = mRotZ.unit();
  mRotY = mRotZ.cross(mRotX); // y'
  mRotY = mRotY.unit();
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GateVSource::InitializeUserFocalShape()
{
  mIsUserFocalShapeActive = true;
  mUserFocalShapeInitialisation = false;
  mUserFocalShape->SetBiasRndm( GetBiasRndm() );
  mUserFocalShape->SetPosDisType("Plane");
  mUserFocalShape->SetPosDisShape("Circle");

  m_angSPS->SetAngDistType("focused");
}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
G4ThreeVector GateVSource::UserFocalShapeGenerateOne()
{
  G4ThreeVector position = mUserFocalShape->GenerateOne();
//   DD(position);
  m_angSPS->SetFocusPoint(position);
  G4ThreeVector momentum = m_angSPS->GenerateOne();
//   DD(momentum);
//   DD(position);
//   DD(momentum);

  return momentum;
}
//----------------------------------------------------------------------------------------
