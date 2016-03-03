/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
/*! \file GateImageDeformActor.cc
    \brief Implementation of GateImageDeformActor
    \author yannick.lemarechal@univ-brest.fr
	    david.sarrut@creatis.insa-lyon.fr
*/

#include "GateImageDeformActor.hh"
#include "GateImageDeformActorMessenger.hh"
#include "G4ThreeVector.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4UnitsTable.hh"
#include "GateTools.hh"
#include "GateMiscFunctions.hh"
#include "GateVVolume.hh"
#include "GateGenericRepeater.hh"
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
//-------------------------------------------------------------------------------------------------
GateImageDeformActor::GateImageDeformActor ( G4String name, G4int depth ) :
    GateVActor ( name,depth ), mMessenger ( 0 )
{
    currentPhase = -1;
    pClock = GateClock::GetInstance();

    GateDebugMessageDec ( "Actor",4,"GateImageDeformActor() -- begin"<<G4endl );
    mMessenger = new GateImageDeformActorMessenger ( this );

    GateDebugMessageDec ( "Actor",4,"GateImageDeformActor() -- end"<<G4endl );

}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateImageDeformActor::~GateImageDeformActor()
{
    delete mMessenger;
    delete pClock;
}
//------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateImageDeformActor::SetFilename ( G4String filename )
{
    
    mPDFFile = filename;
}

/// Construct
void GateImageDeformActor::Construct()
{
    GateMessage ( "Actor", 4, "GateImageDeformActor -- Construct - begin" << G4endl );
    GateVActor::Construct();

    EnableBeginOfRunAction ( true );
    EnableEndOfRunAction ( true ); // for save
    EnableBeginOfEventAction ( false );
    EnableEndOfEventAction ( false ); // for save every n
    readPDFFile ( mPDFFile );

    GateVImageVolume * volAsImage = dynamic_cast<GateVImageVolume*> ( mVolume );
    if ( volAsImage )
    {
    }
    else
    {
        GateError ( "GateImageDeformActor -- It must be a image volume" );
    }

    G4cout<<GetObjectName() <<G4endl;

    
    
    GateMessage ( "Actor", 4, "GateImageDeformActor -- Construct - end" << G4endl );
}


void GateImageDeformActor::readPDFFile ( G4String filename )
{

    GateMessage ( "Actor", 4, "GateImageDeformActor -- Read file " << filename << G4endl );
    
    std::ifstream is ( filename,std::ifstream::in );

    if ( !is )
    {
        GateError ( "GateImageDeformActor -- File " << filename << "does not exist" << G4endl );

    }
    else
    {
        G4int i=0;
        while ( is )
        {
            skipComment ( is );
            double h1;
            G4String h2;

            is >> h1;
            is >> h2;
            mTimeList.push_back ( h1 );
            mFileList.push_back ( h2 );
            ++i;
            if ( is.eof() ) break;

        }
        mNumberOfPhases = i;

    }

    is.close();

}

//-----------------------------------------------------------------------------
void GateImageDeformActor::BeginOfRunAction ( const G4Run * r )
{
    GateVActor::BeginOfRunAction ( r );
    GateMessage ( "Actor", 3, "GateImageDeformActor -- Begin of Run" << G4endl );

    double i=0.;
    G4int phase=0;
    for ( phase=0; phase<mNumberOfPhases; ++phase )
    {
        i =  mTimeList[phase];
        if ( i>=pClock->GetTime() /s ) break;
    }
    
    if ( phase>mNumberOfPhases ) phase = mNumberOfPhases-1; 
    

    if ( phase != currentPhase ) // Change CT densities
    {
	
	GateMessage ( "Actor", 4, "GateImageDeformActor -- Change from phase " << currentPhase << " to " << phase << G4endl );
	
        currentPhase=phase;
	
        GateVImageVolume * volAsImage = dynamic_cast<GateVImageVolume*> ( mVolume );

        if ( volAsImage ) 
        {
            GateImage* tmp = new GateImage;
            tmp->Read ( mFileList[phase] );

	
            if ( ( tmp->GetSize() == volAsImage->GetImage()->GetSize() ) &&
                 ( tmp->GetVoxelSize() == volAsImage->GetImage()->GetVoxelSize() ) &&
                 ( tmp->GetOrigin()    == volAsImage->GetImage()->GetOrigin() ) &&
                 ( tmp->GetTransformMatrix() == volAsImage->GetImage()->GetTransformMatrix() ) ) 
            {

                volAsImage->SetImageFilename ( mFileList[phase] );
                volAsImage->SetImage ( tmp );

                volAsImage->LoadImage ( false );

		volAsImage->LoadImageMaterialsTable();

            }
            else
            {
                GateError ( "GateImageDeformActor -- Volumes must have the same dimension (voxel size, number of voxels, origin and rotation matrix" );

            }

        } // if ( volAsImage )
        else
        {
            G4cout<<"No voxelised volume ... "<<G4endl;
        }

    }

}
//-----------------------------------------------------------------------------
void GateImageDeformActor::EndOfRunAction ( const G4Run * r )
{
    GateVActor::EndOfRunAction ( r );
    GateMessage ( "Actor", 3, "GateImageDeformActor -- End of Run" << G4endl );
    GateMessageDec ( "Actor",4,"\033[32;GateImageDeformActor() -- end"<<G4endl );
}
//-----------------------------------------------------------------------------
// Callback at each event
void GateImageDeformActor::BeginOfEventAction ( const G4Event * e )
{
    GateVActor::BeginOfEventAction ( e );
//   mCurrentEvent++;
    GateDebugMessage ( "Actor", 3, "GateImageDeformActor -- Begin of Event: "<<mCurrentEvent << G4endl );
}
