/*!
 *	\file PET_Analyse.cpp
 *	\author Sebastian Jan <sjan@cea.fr> - March 2007
 *	\author Uwe Pietrzyk <u.pietrzyk@fz-juelich.de> - March 2010
 *
 *	\brief Example of a ROOT C++ code to:
 *		- Read an output root data file
 *		- Create a loop on each event which are stored during the simulation
 *		- Perform data processing
 *		- Plot the results
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TH3.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TApplication.h>

/*!
 *	\fn void printHelpAndQuit( std::string const* msg )
 *	\param msg error message
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
	
	TTree* coincidences = (TTree*)gDirectory->Get( "Coincidences" );

	// Creation of 2 histograms of 1 dimension
	TH1F* gamma1 = new TH1F( "gamma1", "", 80, 0.2, 0.8 );
	TH1F* gamma2 = new TH1F( "gamma2", "", 100, 0.2, 0.8 );

	// Creation of 1 histogram of 3 dimensions
	TH3F* position = new TH3F( "position", "", 200, -400, 400, 200, -400, 400
			, 200, -400, 440 );

	// Declaration of leaves types - TTree coincidences
	Float_t 				axialPos;
	Char_t          comptVolName1[ 40 ];
	Char_t          comptVolName2[ 40 ];
	Int_t           comptonPhantom1;
	Int_t           comptonPhantom2;
	Int_t           comptonCrystal1;
	Int_t           comptonCrystal2;
	Int_t           crystalID1;
	Int_t           crystalID2;
	Float_t         energy1;
	Float_t         energy2;
	Int_t           eventID1;
	Int_t           eventID2;
	Float_t         globalPosX1;
	Float_t         globalPosX2;
	Float_t         globalPosY1;
	Float_t         globalPosY2;
	Float_t         globalPosZ1;
	Float_t         globalPosZ2;
	Int_t           layerID1;
	Int_t           layerID2;
	Int_t           moduleID1;
	Int_t           moduleID2;
	Float_t         rotationAngle;
	Int_t           rsectorID1;
	Int_t           rsectorID2;
	Int_t           runID;
	Float_t         sinogramS;
	Float_t         sinogramTheta;
	Int_t           sourceID1;
	Int_t           sourceID2;
	Float_t         sourcePosX1;
	Float_t         sourcePosX2;
	Float_t         sourcePosY1;
	Float_t         sourcePosY2;
	Float_t         sourcePosZ1;
	Float_t         sourcePosZ2;
	Int_t           submoduleID1;
	Int_t           submoduleID2;
	Double_t        time1;
	Double_t        time2;

	// Set branch addresses - TTree coincidences
	coincidences->SetBranchAddress( "axialPos", &axialPos );
	coincidences->SetBranchAddress( "comptVolName1", &comptVolName1 );
	coincidences->SetBranchAddress( "comptVolName2", &comptVolName2 );
	coincidences->SetBranchAddress( "comptonPhantom1", &comptonPhantom1 );
	coincidences->SetBranchAddress( "comptonPhantom2", &comptonPhantom2 );
	coincidences->SetBranchAddress( "comptonCrystal1", &comptonCrystal1 );
	coincidences->SetBranchAddress( "comptonCrystal2", &comptonCrystal2 );
	coincidences->SetBranchAddress( "crystalID1", &crystalID1 );
	coincidences->SetBranchAddress( "crystalID2", &crystalID2 );
	coincidences->SetBranchAddress( "energy1", &energy1 );
	coincidences->SetBranchAddress( "energy2", &energy2 );
	coincidences->SetBranchAddress( "eventID1", &eventID1 );
	coincidences->SetBranchAddress( "eventID2", &eventID2 );
	coincidences->SetBranchAddress( "globalPosX1", &globalPosX1 );
	coincidences->SetBranchAddress( "globalPosX2", &globalPosX2 );
	coincidences->SetBranchAddress( "globalPosY1", &globalPosY1 );
	coincidences->SetBranchAddress( "globalPosY2", &globalPosY2 );
	coincidences->SetBranchAddress( "globalPosZ1", &globalPosZ1 );
	coincidences->SetBranchAddress( "globalPosZ2", &globalPosZ2 );
	coincidences->SetBranchAddress( "layerID1", &layerID1 );
	coincidences->SetBranchAddress( "layerID2", &layerID2 );
	coincidences->SetBranchAddress( "moduleID1", &moduleID1 );
	coincidences->SetBranchAddress( "moduleID2", &moduleID2 );
	coincidences->SetBranchAddress( "rotationAngle", &rotationAngle );
	coincidences->SetBranchAddress( "rsectorID1", &rsectorID1 );
	coincidences->SetBranchAddress( "rsectorID2", &rsectorID2 );
	coincidences->SetBranchAddress( "runID", &runID );
	coincidences->SetBranchAddress( "sinogramS", &sinogramS );
	coincidences->SetBranchAddress( "sinogramTheta", &sinogramTheta );
	coincidences->SetBranchAddress( "sourceID1", &sourceID1 );
	coincidences->SetBranchAddress( "sourceID2", &sourceID2 );
	coincidences->SetBranchAddress( "sourcePosX1", &sourcePosX1 );
	coincidences->SetBranchAddress( "sourcePosX2", &sourcePosX2 );
	coincidences->SetBranchAddress( "sourcePosY1", &sourcePosY1 );
	coincidences->SetBranchAddress( "sourcePosY2", &sourcePosY2 );
	coincidences->SetBranchAddress( "sourcePosZ1", &sourcePosZ1 );
	coincidences->SetBranchAddress( "sourcePosZ2", &sourcePosZ2 );
	coincidences->SetBranchAddress( "submoduleID1", &submoduleID1 );
	coincidences->SetBranchAddress( "submoduleID2", &submoduleID2 );
	coincidences->SetBranchAddress( "time1", &time1 );
	coincidences->SetBranchAddress( "time2", &time2 );

	Int_t nentries = coincidences->GetEntries();
	std::cout << "nentries: " << nentries << std::endl;

	Int_t nbytes( 0 );

	// Loop on the events in the TTree Coincidences
	Int_t nbrCoincPrompt( 0 );
	Int_t nbrCoincRandom( 0 );
	Int_t nbrCoincScatter( 0 );
	Int_t nbrCoincTrues( 0 );
	Float_t nTot( 10000000.0 );
	Float_t sensi( 0.0 );

	Int_t i( 0 );
	while( i < nentries )
	{
		nbytes += coincidences->GetEntry( i );
		
		// Fill gamma1 histo without condition
		gamma1->Fill( energy1 );

		// Fill the gamma2 histo with condition
		if( energy2 >= 0.4 )
		{
			gamma2->Fill( energy2 );
		}

		// Fill the 3D Histo without condition
		position->Fill( globalPosZ1, globalPosX1, globalPosY1 );		

		/***********************************
		 *																 *
		 * EVALUATION OF:                  *
		 *	- PROMPTS											 *
		 *	- TRUES												 *
		 *	- RANDOM											 *
		 *	-	SENSITIVITY									 *
		 *																 *
		 ***********************************/
		
		++nbrCoincPrompt;

		if( eventID1 != eventID2 )
		{
			++nbrCoincRandom;
		}

		if( eventID1 == eventID2 && comptonPhantom1 == 0 
				&& comptonPhantom2 == 0 )
		{
			++nbrCoincTrues;
		}

		++i;
	}

	sensi = nbrCoincPrompt / nTot * 100.0;

	std::cout << std::endl;
	std::cout << std::endl;		
	std::cout << std::endl;
	std::cout << "---> PROMPTS = " << nbrCoincPrompt << " Cps" << std::endl;
	std::cout << "---> TRUES   = " << nbrCoincTrues  << " Cps" << std::endl;
	std::cout << "---> RANDOMS = " << nbrCoincRandom << " Cps" << std::endl;
	std::cout << " ___________________  " << std::endl;
	std::cout << "|                     " << std::endl;
	std::cout << "| TOTAL SENSITIVITY : " << std::showpoint 
						<< std::fixed << std::setprecision( 8 ) 
						<< sensi << " %"<< std::endl;
	std::cout << "| ------------------  " << std::endl;
	std::cout << "|___________________  " << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
		
	// Plot the results

	gStyle->SetPalette( 1 );
		
	TCanvas* c1 = new TCanvas( "Reco", "Reco", 200, 10, 500, 600);
	c1->SetFillColor( 0 );
	c1->SetBorderMode( 0 );

	gamma1->Draw();
	gamma1->SetFillColor( 2 );
	
	gamma2->Draw( "same" );
	gamma2->SetFillColor( 9 );
  
	TLatex* tex = new TLatex( 0.255919, 600.0, "GAMMA 1" );
	tex->SetTextColor( 2 );
	tex->SetTextSize( 0.05 );
	tex->SetLineWidth(2);
	tex->Draw();
	
	TLatex* tex2 = new TLatex( 0.620151, 300.0, "GAMMA 2" );
	tex2->SetTextColor( 9 );
	tex2->SetTextSize( 0.05 );
	tex2->SetLineWidth( 2 );
	tex2->Draw();
    
	c1->Update();

	TCanvas* c2 = new TCanvas("Reco_true","Reco_true",200,10,500,600);
	c2->SetFillColor(0);
	c2->SetGrid();
	c2->SetBorderMode(0);

	position->Draw();
 
	c2->Update();

	theApp.Run();

	delete tex;
	delete tex2;
	delete position;
	delete gamma2;	
	delete gamma1;
	delete coincidences;
	delete f;

	return 0;
}
