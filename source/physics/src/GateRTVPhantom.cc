#include "GateRTVPhantom.hh"      
#include <cstdio>
#include <cmath>

#include <sstream>
#include <iostream>
#include <fstream>

#include "G4GeometryManager.hh"
#include "G4RunManager.hh"
#include "GateRTVPhantomMessenger.hh"
#include "GateVGeometryVoxelReader.hh"
#include "GateVSourceVoxelReader.hh"

#include "GateVoxelCompressor.hh"

#include "GateVVolume.hh"
#include "GateCompressedVoxelParam.hh"
#include "GateObjectStore.hh"
//#include "GateSourceMgr.hh"
#include "GateDetectorConstruction.hh"

 GateRTVPhantom::GateRTVPhantom():GateRTPhantom("RTVPhantom")
{

G4cout << " creating a RTV Phantom at Address "<< this <<G4endl;

    IsVoxellized  = 1;
    IsEnabled     = 1;
    IsInitialized = 0;
    set_AttAsAct = 0;
    set_ActAsAtt = 0;
    cK = 1;
    p_cK = 1;

  XDIM = 50;
  YDIM = 50;
  ZDIM = 50;
  ZDIM_OUTPUT = 50;
  pixel_width = G4ThreeVector(1.,1.,1.);


NbOfFrames =0;

m_TPF = 0.* s;

base_FN    = G4String("NotDefined") ;
current_FN = G4String("NotDefined");
header_FN = G4String("NotDefined");
m_messenger = new GateRTVPhantomMessenger(this);

}

G4double  GateRTVPhantom::GetTPF()
{ return m_TPF; }

void GateRTVPhantom::SetTPF( G4double aTPF)
{m_TPF = aTPF;}

G4int GateRTVPhantom::GetNbOfFrames()
{ return NbOfFrames;}

void   GateRTVPhantom::SetNbOfFrames( G4int aNb )
{ NbOfFrames = aNb;}

void   GateRTVPhantom::SetBaseFileName( G4String aFN ) 
{ base_FN = aFN;
  if ( header_FN == G4String("NotDefined") )
  {
   G4cout<<"GateRTVPhantom::SetBaseFileName ERROR : No header file name has been provided for the attenuation map.";
   G4Exception( "GateRTVPhantom::SetBaseFileName", "SetBaseFileName", FatalException, " Please set it before setting the base file name. Aborting.");
  }
  current_FN = base_FN+"_atn_1.bin";
  
  if ( set_AttAsAct == 1 ) current_FN = base_FN+"_act_1.bin";
  
  itsGReader->ReadRTFile( header_FN, current_FN );
  XDIM = itsGReader->GetVoxelNx();
  YDIM = itsGReader->GetVoxelNy();
  ZDIM_OUTPUT = itsGReader->GetVoxelNz();
  pixel_width = itsGReader->GetVoxelSize() * cm; 
  IsInitialized = 1;
}

void   GateRTVPhantom::SetHeaderFileName( G4String aFN )
{
header_FN = aFN;

G4cout << " GateRTVPhantom::SetHeaderFileName ::: header file name = " <<header_FN<<G4endl;
}

void GateRTVPhantom::Compute(G4double aTime)
{
  static G4bool IsFirstTime = true;
  
 if ( GetNbOfFrames() == 0 ) { G4Exception( "GateRTVPhantom::Compute", "Compute", FatalException, "ERROR  the Number of Frames is set to 0.");}

     G4double time_s = aTime/s;

cK = 1;


if ( GetNbOfFrames() > 1 )
{
  cK = (G4int)( floor( aTime / GetTPF() ) ) + 1;
  cK = cK % GetNbOfFrames(); // get cK modulo the number of frames
}

if ( cK == 0 ) { cK = 1; }
std::stringstream st;
st << cK;

if (  cK != p_cK  && cK <= GetNbOfFrames()  ) 
{

       if ( IsFirstTime == true )
        { 
         if ( itsSReader->GetTimeSampling() < 1e-8 ) // if time sampling for time activity curves is too small
           {
            G4cout << "GateRTVPhantom::Compute  WARNING : Time Sampling for Time Activity Curves is too small - setting default to Time Per Frame."<< G4endl;                                       
            itsSReader->SetTimeSampling( GetTPF() ); // set it by default to TimePerFrame 
           }
        }

/*
G4double TATPF = itsSReader->GetTimeSampling()/s;
G4int newIndex = (G4int)( floor( time_s / TATPF ) ) + 1;
itsSReader->SetFirstIndex ( newIndex );
*/
        if (GetVerboseLevel()>0)
        {
      G4cout << " time is " << time_s << " (s)   -    interval boundaries are (s) " <<  ((cK-1) * GetTPF()/s) << "    " << (cK * GetTPF()/s ) << G4endl;
      G4cout << " in GateSourceMgr::PrepareNextEvent - Time is in new Interval - Updating NCATPhantom Matrix ..." << G4endl;
      G4cout << " index for computing Voxels Matrix = " << cK << G4endl;
        }

// here we load the cKth phantom frame from file

// convert cK into string

if ( set_AttAsAct == 1 ) current_FN = base_FN+"_act_"+st.str()+".bin";
else current_FN = base_FN+"_atn_"+st.str()+".bin";

itsGReader->ReadRTFile( header_FN, current_FN );

XDIM = itsGReader->GetVoxelNx();
YDIM = itsGReader->GetVoxelNy();
ZDIM_OUTPUT = itsGReader->GetVoxelNz();
pixel_width = itsGReader->GetVoxelSize();

//G4cout << " GateRTVPhantom::Compute  AFTER GReader->ReadFile( header_FN, current_FN ) " << XDIM<<" "<<YDIM<<" "<<ZDIM_OUTPUT<<G4endl;

// Destroy and reconstruct physical volumes of enclosing box
//
// rebuild all the G4VoxelsHeaders for the physical volume  enclosing the NCAT phantom : this is COMPULSORY for G4 NAVIGATION
//


if ( G4GeometryManager::GetInstance()->IsGeometryClosed() == false )
 {G4cout << " Destroying Geometry of " << m_inserter->GetObjectName()<<G4endl;
  m_inserter->DestroyGeometry();
  //m_inserter->ConstructGeometry( m_inserter->GetMotherLogicalVolume() , false);
  m_inserter->Construct(false);
 }
 else 
     {G4cout << " Destroying Geometry of " << m_inserter->GetObjectName()<<G4endl;
      G4GeometryManager::GetInstance()->OpenGeometry();
      m_inserter->DestroyGeometry();
      //m_inserter->ConstructGeometry( m_inserter->GetMotherLogicalVolume() , false);
	  m_inserter->Construct(false);
      G4GeometryManager::GetInstance()->CloseGeometry( true, true );
     }


/*
G4cout << " #################### Destroying Geometry of " << GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume()->GetName()<<G4endl;
G4GeometryManager::GetInstance()->OpenGeometry(GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume() );
GateDetectorConstruction::GetGateDetectorConstruction()->GeometryHasChanged(GateDetectorConstruction::geometry_needs_rebuild);
G4GeometryManager::GetInstance()->CloseGeometry( true, true, GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume() );
G4cout << " #################### Geometry of " << GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume()->GetName()<<"has been CLOSED" <<G4endl;

G4cout << " #################### Destroying Geometry of " << m_inserter->GetPhysicalVolume(0)->GetName()<<G4endl;
G4GeometryManager::GetInstance()->OpenGeometry(m_inserter->GetPhysicalVolume(0));
m_inserter->DestroyGeometry();
m_inserter->ConstructGeometry( m_inserter->GetMotherLogicalVolume() , false);
G4GeometryManager::GetInstance()->CloseGeometry( false, true, m_inserter->GetPhysicalVolume(0) );
G4cout << " #################### REBUILT Geometry of " << m_inserter->GetPhysicalVolume(0)->GetName()<<G4endl;
*/

//G4RunManager::GetRunManager()->GeometryHasBeenModified();

}

if ( IsFirstTime == true || (  cK != p_cK  && cK <= GetNbOfFrames()  ) )
{
if ( set_ActAsAtt == 1 ) current_FN = base_FN+"_atn_"+st.str()+".bin";
else current_FN = base_FN+"_act_"+st.str()+".bin";                     
itsSReader->ReadRTFile( header_FN, current_FN );
itsSReader->Dump(0);
IsFirstTime = false;
}

p_cK = cK;

//G4cout << " GateRTVPhantom  :::: UPDATING ACTIVITIES " << G4endl;
//if ( fabs( GetTPF() - itsSReader->GetTimeSampling() ) > 1e-8 ) 
itsSReader->UpdateActivities( header_FN, current_FN );

//G4cout <<" GateRTVPhantom::Compute --- leaving " <<G4endl;

}



