/*!	\file VoxelPhantom_Analyse.cpp
 *  \author Sebastian Jan <sjan@cea.fr> - March 2007
 *	\author Uwe Pietrzyk <u.pietrzyk@fz-juelich.de> - March 2010
 *
 *  \brief Example of a ROOT C++ code to:
 *    - Read an output root data file
 *    - Create a loop on each event which are stored during the simulation
 *    - Perform data processing
 *    - Plot the results
 */
 
#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
 
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TApplication.h>
 
/*!
 *  \fn void printHelpAndQuit( std::string const* msg )
 *  \param msg error message
 */
void printHelpAndQuit( std::string const* msg )
{
	std::cerr << *msg << std::endl;
	std::cerr << "Usage: PET_Analyse <root>" << std::endl;
	std::cerr << "<root> root file to analyse" << std::endl;
	exit( EXIT_FAILURE );
}
 
int main( int argc, char** argv )
{
	if( argc < 2 )
	{
		std::string const msg = "Arguments missing!!!";
		printHelpAndQuit( &msg );
	}
 
	gROOT->Reset();

	// Open the root file and check if it exists
	std::string const inputRootFile( argv[ 1 ] );
 	TFile* f = new TFile( inputRootFile.c_str(), "READ" );
	if( f->IsZombie() )
	{
		std::cerr << "Cannot open the file '" << inputRootFile
			        << "'" << std::endl;
		exit( EXIT_FAILURE );
	}
 
	TApplication theApp( "Application", &argc, argv );

	TTree* singles = (TTree*)gDirectory->Get( "Singles" );

	// STAT   
	gStyle->SetStatW( 0.28 );
	gStyle->SetStatH( 0.13 );
	gStyle->SetStatColor( 41 );
	gStyle->SetStatX( 0.87 );
	gStyle->SetStatY( 0.85 );
	gStyle->SetStatFont( 42 );
	gStyle->SetOptStat( 111 );
 
	// Creation of histo 2 Dim. 
	TH2F* phantom = new TH2F( "Phantom", "", 100, -100, 100, 100, -100, 100 );
 
	//Declaration of leaves types - TTree Singles
	Float_t sourcePosX;
	Float_t sourcePosY;
	Float_t sourcePosZ;

	//Set branch addresses - TTree Singles
	singles->SetBranchAddress( "sourcePosX", &sourcePosX );
	singles->SetBranchAddress( "sourcePosY", &sourcePosY );
	singles->SetBranchAddress( "sourcePosZ", &sourcePosZ );

	Int_t nentries = singles->GetEntries();
	Int_t nbytes( 0 );

	//Loop on event number for Singles TTree
	Int_t i( 0 );
	while( i < nentries )
	{
		nbytes += singles->GetEntry( i );
		phantom->Fill( sourcePosX, sourcePosY );
		++i;
	}

	// Result plots
	gStyle->SetPalette( 1 );
 
	TCanvas* c1 = new TCanvas( "Reco", "Reco", 200, 10, 500, 600 );
	c1->SetFillColor( 0 );
	c1->SetBorderMode( 0 );
 
	phantom->Draw( "contZ" );

	c1->Update();

	theApp.Run();

	delete c1;
	delete singles;
	delete phantom;
	delete f;

	return 0;
}
