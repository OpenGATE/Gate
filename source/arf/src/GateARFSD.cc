/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFSD.hh"
#include "GateCrystalHit.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ios.hh"
#include "G4VProcess.hh"

#include "G4TransportationManager.hh"

#include "GateVSystem.hh"
#include "GateRotationMove.hh"
#include "GateOrbitingMove.hh"
#include "GateEccentRotMove.hh"
#include "GateSystemListManager.hh"
#include "GateVVolume.hh"
#include "GateDigitizer.hh"

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

#include "globals.hh"

#include "GateARFSDMessenger.hh"

#include "GateARFTableMgr.hh" // manages the ARF tables

#include "GateVVolume.hh"
#include "G4ThreeVector.hh"
#include <ctime>
#include "GateBox.hh"
#include "GateToProjectionSet.hh"
#include "GateOutputMgr.hh"
#include "TH1D.h"

// Name of the hit collection
const G4String GateARFSD::theARFCollectionName = "ARFCollection";




// Constructor
GateARFSD::GateARFSD(const G4String& pathname, G4String aName)
:G4VSensitiveDetector(pathname),m_system(0),m_name(aName)
{
  collectionName.insert(theARFCollectionName);

 m_NbOfRejectedPhotons = 0;

m_messenger = new GateARFSDMessenger(this);

nbofGoingIn = 0;

 m_inserter = 0;

theProjectionSet = 0;

m_ARFTableMgr = new GateARFTableMgr( GetName(), this );

  m_file = 0;
  m_singlesTree = 0;
  NbOfSimuPhotons = 0;
  NbofGoingOutPhotons = 0;
  NbofStraightPhotons = 0;
  NbofGoingInPhotons = 0;
  NbOfSourcePhotons = 0;
  NbOfGoodPhotons=0;
  NbofStoredPhotons = 0;
  NbOfHeads = 0;
m_edepthreshold = 0.;
headID = -1;
m_XPlane =0.;
m_ARFStage = -2;
G4cout << " created a ARF Sensivitive Detector "<< G4endl;
}


// Destructor
GateARFSD::~GateARFSD()
                       { delete m_messenger;
                         delete m_ARFTableMgr;
                       }

// Method overloading the virtual method Initialize() of G4VSensitiveDetector
void GateARFSD::Initialize(G4HCofThisEvent*HCE)
{
  static int HCID = -1; // Static variable storing the hit collection ID

  // Creation of a new hit collection
  ARFCollection = new GateCrystalHitsCollection
                   (SensitiveDetectorName,theARFCollectionName); 

  // We store the hit collection ID into the static variable HCID
  if(HCID<0)
  { HCID = GetCollectionID(0); }
 
  // Add the hit collection to the G4HCofThisEvent
  HCE->AddHitsCollection(HCID,ARFCollection);
}




// Implementation of the pure virtual method ProcessHits().
// This methods generates a GateGeomColliHit and stores it into the SD's hit collection
//G4bool GateGeomColliSD::ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist)


G4bool GateARFSD::ProcessHits(G4Step*aStep,G4TouchableHistory*)
{
G4Track* theTrack = static_cast<G4Track*>( aStep->GetTrack() );


    
if ( theTrack->GetParentID() != 0 ) return false;
theTrack->SetTrackStatus( fKillTrackAndSecondaries );
G4ThreeVector thePosAtVertex = theTrack->GetVertexPosition();
     
G4ThreeVector thePosition = theTrack->GetPosition();
G4double theInEnergy = theTrack->GetTotalEnergy();

  // Get the step-points
  G4StepPoint  *oldStepPoint = aStep->GetPreStepPoint(),
      	       *newStepPoint = aStep->GetPostStepPoint();

  const G4VProcess*process = newStepPoint->GetProcessDefinedStep();
 
   //  For all processes except transportation, we select the PostStepPoint volume
   //  For the transportation, we select the PreStepPoint volume
   const G4TouchableHistory* touchable;
   if ( process->GetProcessType() == fTransportation )
      touchable = (const G4TouchableHistory*)(oldStepPoint->GetTouchable() );
  else 
      touchable = (const G4TouchableHistory*)(newStepPoint->GetTouchable() );
  GateVolumeID volumeID(touchable);

  headID = volumeID.GetVolume( volumeID.GetCreatorDepth("SPECThead") )->GetCopyNo();
  

if (volumeID.IsInvalid()) //G4Exception("[GateARFSD]: could not get the volume ID! Aborting!\n");
{
	G4Exception( "GateARFSD::ProcessHits", "ProcessHits", FatalException, "Could not get the volume ID! Aborting!" );
}

// now we compute the position in the current frame to be able to extract the angles theta and phi


    G4ThreeVector localPosition = volumeID.MoveToBottomVolumeFrame( thePosition );
    G4ThreeVector VertexPosition = volumeID.MoveToBottomVolumeFrame( thePosAtVertex );
    G4ThreeVector theDirection = localPosition - VertexPosition;
    G4double mag = theDirection.mag();
    theDirection /= mag;

//    G4cout << " local position  " << localPosition << G4endl;
//    G4cout << " Vertex Position "<<VertexPosition<< G4endl;
//    G4cout << " direction      " << theDirection<<G4endl;

    ComputeProjectionSet(localPosition, theDirection , theInEnergy );


return true;
}

G4int GateARFSD::PrepareCreatorAttachment(GateVVolume* aCreator) 
{ 
  GateVSystem* creatorSystem = GateSystemListManager::GetInstance()->FindSystemOfCreator(aCreator);
  if (!creatorSystem) {
    G4cout  << G4endl << G4endl << "[GateARFSD::PrepareCreatorAttachment]:" << G4endl
     << "Volume '" << aCreator->GetObjectName() << "' does not belong to any system." << G4endl
           << "Your volume must belong to a system to be used with a GeomColliSD." << G4endl
     << "Attachment request ignored --> you won't have any hit output from this volume!!!" << G4endl << G4endl;
    return -1;
  }

  if (m_system) {
    if (creatorSystem!=m_system)     {
      G4cout  << G4endl << G4endl << "[GateARFSD::PrepareCreatorAttachment]:" << G4endl
       << "Volume '" << aCreator->GetObjectName() << "' belongs to system '" << creatorSystem->GetObjectName() << "'" << G4endl
             << "while the GeomColliSD has already been attached to a volume from another system ('" << m_system->GetObjectName()<< "')." << G4endl
       << "Attachment request ignored --> you won't have any hit output from this volume!!!" << G4endl << G4endl;
      return -1;
    }
  }
  else
      SetSystem(creatorSystem);
  
  return 0;
}


// Set the system to which the SD is attached
void GateARFSD::SetSystem(GateVSystem* aSystem)
{ 
  m_system=aSystem;
  GateDigitizer::GetInstance()->SetSystem(aSystem);
}

void GateARFSD::computeTables()
{ // open the root files generated from the ARF simu

if ( m_ARFStage != 1 )
{
	G4Exception( "GateARFSD::computeTable", "computeTable", FatalException, "Illegal state of the Gate ARF Sensitive Detector" );
}

G4cout << "GateARFSD::computeTables() -  Computing ARF Tables for Sensitive Detector " << GetName() << G4endl;

      time_t theTimeBefore = time(NULL);


      G4int loaded = m_ARFTableMgr->InitializeTables();
  if ( loaded == 1 ) return;

               std::map<G4String,G4int>::iterator iter; 
    G4double* NSourcePhotons = new G4double[ m_EnWin.size() ];
    G4int iw = 0;
    for (iter= m_EnWin.begin(); iter != m_EnWin.end(); iter++ )
     {              NbOfSimuPhotons = 0;
                    NbofGoingOutPhotons =0 ;
                    NbofStraightPhotons =0 ;
                    NbofGoingInPhotons= 0;
                    NbOfSourcePhotons = 0;
                    NbofStoredPhotons = 0;
		    IN_camera = 0;
		    OUT_camera = 0;
                   G4int TotNbOfSingles = 0;
                   G4String cfn = (iter->first) +".root";
                   ULong64_t NbofGoingOutPhotons_tmp = 0;
                   ULong64_t NbofGoingInPhotons_tmp = 0;
                   ULong64_t NbOfSourcePhotons_tmp = 0;
                   ULong64_t NbofStoredPhotons_tmp = 0;
                   ULong64_t IN_camera_tmp = 0;
                   ULong64_t OUT_camera_tmp = 0;
                   
                   for( G4int i = 0 ; i < ( iter->second ) ; i++ )
                   {
                    if ( i > 0 ) {
                                    std::stringstream s;
                                    s << i;
		                    cfn = (iter->first) + "_"+s.str()+".root"; }

                    if ( m_file != 0 ) { delete m_file; m_file = 0;}

                    m_file = new TFile( cfn.c_str() ,"READ","ROOT filefor ARF purpose");
                    G4cout << "GateARFSD::computeTables():::::: Reading ROOT File  " << cfn << G4endl;
                    m_singlesTree = (TTree* ) ( m_file->Get("theTree") );
                    G4cout << " m_singlesTree = " << m_singlesTree<<G4endl;
                    m_singlesTree->SetBranchAddress("Edep", &theData.m_Edep);
   		            m_singlesTree->SetBranchAddress("outY", &theData.m_Y);
   		            m_singlesTree->SetBranchAddress("outX", &theData.m_X);
                    m_NbOfPhotonsTree = (TTree* ) ( m_file->Get("theNumberOfPhoton") );
                    G4cout << " m_NbOfPhotonsTree = " <<m_NbOfPhotonsTree<< G4endl;
                    m_NbOfPhotonsTree->SetBranchAddress("NOfOutGoingPhot",&NbofGoingOutPhotons_tmp);
                    m_NbOfPhotonsTree->SetBranchAddress("NbOfInGoingPhot",&NbofGoingInPhotons_tmp);
                    m_NbOfPhotonsTree->SetBranchAddress("NbOfSourcePhot",&NbOfSourcePhotons_tmp);
                    m_NbOfPhotonsTree->SetBranchAddress("NbOfStoredPhotons",&NbofStoredPhotons_tmp);
                    m_NbOfPhotonsTree->SetBranchAddress("NbOfInCameraPhot",&IN_camera_tmp);
                    m_NbOfPhotonsTree->SetBranchAddress("NbOfOutCameraPhot",&OUT_camera_tmp);
                    m_NbOfPhotonsTree->SetBranchAddress("NbOfHeads",&NbOfHeads);
                    m_NbOfPhotonsTree->GetEntry(0);

                    NbofGoingOutPhotons += NbofGoingOutPhotons_tmp;
                    NbofGoingInPhotons += NbofGoingInPhotons_tmp;
                    NbOfSourcePhotons += NbOfSourcePhotons_tmp;
                    NbofStoredPhotons += NbofStoredPhotons_tmp;
 		    IN_camera += IN_camera_tmp;
 		    OUT_camera += OUT_camera_tmp;
 		    
                    G4cout << " In File " << cfn << G4endl;
                    G4cout << " Total number of Source photons Going Out Crystal  " << NbofGoingOutPhotons_tmp<<G4endl;
                    G4cout << " Total number of Source photons Going In Crystal   " << NbofGoingInPhotons_tmp<<G4endl;
                    G4cout << " Total number of Source photons                    " << NbOfSourcePhotons_tmp<<G4endl;
                    G4cout << " Total number of Source photons Going In Camera    " << IN_camera_tmp<<G4endl;
                    G4cout << " Total number of Source photons Going Out Camera   " << OUT_camera_tmp<<G4endl;
                    G4cout << " Total number of Stored photons                    " << NbofStoredPhotons_tmp<<G4endl;
                    TotNbOfSingles = m_singlesTree->GetEntries();
                    G4cout << " File " << cfn << " contains " << TotNbOfSingles << " entries " << G4endl;
                    G4cout << " Tree m_NbOfPhotonsTree " << cfn << " contains " << m_NbOfPhotonsTree->GetEntries() << " entries " << G4endl;
                    for ( G4int j = 0 ;  j <  TotNbOfSingles ; j++ )
                    {
                     m_singlesTree->GetEntry(j);
                     // loop through ARF tables to get the table with the suitable energy window

                     if ( theData.m_Edep/keV - m_edepthreshold >= 0. )
                      {                      	
                       m_ARFTableMgr->FillDRFTable(iw , theData.m_Edep,  theData.m_X , theData.m_Y );
                      }
                    }
                    m_file->Close();
                   }
                  time_t theTimeAfter = time(NULL);
                  NSourcePhotons[iw] = NbOfSourcePhotons * NbOfHeads;
                  G4cout << " ARF Table # "<<iw<<"  Computation Time " << (theTimeAfter - theTimeBefore ) << " seconds " << G4endl;
                  //G4cout << " the Total final number of photons simulated          " << NbOfSimuPhotons << G4endl;
                  //G4cout << " number of Energy Windows " << m_EnWin.size() <<G4endl;
                  //G4cout << " Number of SOURCE photons                             " << NbOfSourcePhotons << G4endl;
                  //G4cout << " --- Of Which : Number of Photons Going Out Crystal   " << NbofGoingOutPhotons << " = " << 100. * double(NbofGoingOutPhotons)/double(NbOfSourcePhotons) << " %"<<G4endl;
                  //G4cout << " --- Of Which : Number of Photons Going IN Crystal    " << NbofGoingInPhotons << " = " << 100. * double(NbofGoingInPhotons)/double(NbOfSourcePhotons) << " %"<<G4endl;
                  //G4cout << " --- Of Which : Number of Photons Going Out Camera    " << OUT_camera << " = " << 100. * double(OUT_camera)/double(NbOfSourcePhotons) << " %"<<G4endl;
                  //G4cout << " --- Of Which : Number of Photons Going IN Camera     " << IN_camera << " = " << 100. * double(IN_camera)/double(NbOfSourcePhotons) << " %"<<G4endl;
                  //G4cout << " --- Of Which : Number of Stored Photons              " << NbofStoredPhotons<< " = " << 100. * double(NbofStoredPhotons)/double(NbOfSourcePhotons) << " %"<<G4endl;
                  iw++; // now for next ARF table

        }   

      //G4cout << " Number of Heads " << NbOfHeads<<G4endl;

   m_ARFTableMgr->SetNSimuPhotons( NSourcePhotons );

      m_ARFTableMgr->convertDRF2ARF();

      }


void GateARFSD::ComputeProjectionSet(G4ThreeVector thePosition,G4ThreeVector theDirection,G4double theEnergy)
{

 nbofGoingIn++;

//     transform to the detector frame the photon position

//
// we compute the direction and position relative to the detector frame
// we store also the rotation matrix and the translation of the detector relative to the world frame
//
// a position and aDirection are computed relative to the detector frame !
//

// the coordinates of the intersection of the path of the photon with the back surface of the detector
// is given by
//             x = deltaX/2
//             y = yin + t * uy
//             z = zin + t * uz
//
// where
//          u(ux,uy,uz) is the direction vector of the photon
//          (xin,yin,zin) is the starting  position of the photon when it enters the detector
// and
//
//            t = ( deltaX - xin ) / ux
//            deltaX is the projection plane of the detector on the Ox axis
//
//
//           all these coordinates are relaitve to the detector frame where the origin of hte detector is a t the center


G4double theARFvalue = m_ARFTableMgr->ScanTables(  theDirection.z() , theDirection.y() , theEnergy);

//if (theARFvalue>0.1)G4cout << acos(costheta) << "   " << atan(tanphi) << "   "<<theARFvalue<<G4endl;

// the coordinates of the intersection of the path of the photon with the back surface of the detector
// is given by
//             x = deltaX/2
//             y = yin + t * uy
//             z = zin + t * uz
//
// where
//          u(ux,uy,uz) is the direction vector of the photon
//          (xin,yin,zin) is the starting  position of the photon when it enters the detector
// and
//
//            t = ( deltaX/2 - xin ) / ux
//            deltaX is the dimension of the detector on the Ox axis
//
//
//           all these coordinates are relaitve to the detector frame where the origin of hte detector is a t the center

              G4double t = ( m_XPlane - thePosition.x() )/theDirection.x() ;
              G4double xp = thePosition.z() + t * theDirection.z();
              G4double yp = thePosition.y() + t * theDirection.y();

// now store projection with the GateProjectionSet Module thourgh its method GateProjectionSet::Fill


if ( theProjectionSet == 0 ) {GateOutputMgr* outputMgr = GateOutputMgr::GetInstance(); 
                              GateToProjectionSet* PSet = dynamic_cast<GateToProjectionSet*>( outputMgr->GetModule("projection") );
                              if ( PSet == 0 ) { G4Exception( "GateARFSD::ComputeProjectionSet()", "ComputeProjectionSet", FatalException, "ERROR No Projection Set Module has been enabled. Aborting.");}
                              theProjectionSet = PSet->GetProjectionSet();
                             }

// G4cout << " BINNING PROJECTION FOR HEAD ID " << headID << " ( "<<xp<<" ; "<<yp<<") " << G4endl;

theProjectionSet->FillARF( headID , xp , yp , theARFvalue );


}


#endif
