#include <iostream>
#include <cerrno>
#include <cstdlib>
#include <set>
#include <cmath>

#include "TApplication.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TPad.h"

using namespace std;

int main( int argc, char* argv[] )
{
	if( argc < 2 )
	{
		cerr << "arguments missing" << endl;
		cerr << "Usage : AnalyzeCT myFile.root " << endl;
		exit( EXIT_FAILURE );
	}

	// Store the root file name in 'fileName' variable
	char* const FILENAME = argv[ 1 ];

	// parameters for the histograms
	// Size and number of pixel (in micron)
	Double_t const PIXELSIZE = 0.5;
	Int_t const RAW = 100;
	Int_t const COLUMN = 100;
	// Energy bounds in MeV
	Float_t const EMIN = 0.01;
	Float_t const EMAX = 0.04;

	TApplication app( "Application", &argc, argv );

	// Create and initialize a canvas
	TCanvas* canvas = new TCanvas( "Canvas BenchmarkCT", "BenchmarkCT",
			200, 20, 1000, 700 );
	canvas->SetFillColor( 29 );
	canvas->ToggleToolBar();
	canvas->ToggleEventStatus();

	// Open (check) and read the root file
	TFile* file = new TFile( FILENAME );
	if( !file->IsOpen() )
	{
		cerr << "problem opening the root file : '" << FILENAME << "'" << endl;
		cerr << strerror( errno ) << endl;
		exit( EXIT_FAILURE );
	}

	// Take the single tree, where is the position, the energy and the runID
	TTree* singlesTree = (TTree*)file->Get( "Singles" );
	// Global Position in X, Y and Z
	Float_t globalPosX, globalPosY, globalPosZ, energy;
	Int_t runID, pixelID;
	singlesTree->SetBranchAddress( "globalPosX", &globalPosX );
	singlesTree->SetBranchAddress( "globalPosY", &globalPosY );
	singlesTree->SetBranchAddress( "globalPosZ", &globalPosZ );
	singlesTree->SetBranchAddress( "energy", &energy );
	singlesTree->SetBranchAddress( "runID", &runID );
	singlesTree->SetBranchAddress( "pixelID", &pixelID );

	// Number of entries in the single tree
	Int_t entriesSingleTree = (Int_t)singlesTree->GetEntries();
	cout << "Number of detected photons : " << entriesSingleTree << endl;

	// Number of generated photons during the benchmarkCT simulation
	TTree* gateTree = (TTree*)file->Get( "Gate" );

	// Create histogram for each run (2 runs during this benchmarkCT)
	// Define the bounds of the histogram
	Double_t const RAW_BOUND =  PIXELSIZE * RAW / 2;
	Double_t const COLUMN_BOUND = PIXELSIZE * COLUMN / 2;
	TH2F* run_0 = new TH2F( "runID = 0", "projection during the first run",
			COLUMN, -COLUMN_BOUND, COLUMN_BOUND,
			RAW, -RAW_BOUND, RAW_BOUND );
	TH2F* run_1 = new TH2F( "runID = 1", "projection during the second run",
			COLUMN, -COLUMN_BOUND, COLUMN_BOUND,
			RAW, -RAW_BOUND, RAW_BOUND );

	// RunID 0
	// Create histogram for energy spectrum at different position
	Int_t const BIN = 80;
	TH1F* energyRun0Area1 = new TH1F( "Energy RunID = 0, Area 1",
			"energy spectrum : outside of the phantom", BIN, EMIN, EMAX );
	TH1F* energyRun0Area2 = new TH1F( "Energy RunID = 0, Area 2",
			"energy spectrum : in water cylinder", BIN, EMIN, EMAX );
	TH1F* energyRun0Area3 = new TH1F( "Energy RunID = 0, Area 3",
			"energy spectrum : behind the 2 top balls", BIN, EMIN, EMAX );
	TH1F* energyRun0Area4 = new TH1F( "Energy RunID = 0, Area 4",
			"energy spectrum : behind the 2 bottom balls", BIN, EMIN, EMAX );

	// RunID 1
	TH1F* energyRun1Area1 = new TH1F( "Energy RunID = 1, Area 1",
			"energy spectrum : behind the top left ball (PVC)",
			BIN, EMIN, EMAX );
	TH1F* energyRun1Area2 = new TH1F( "Energy RunID = 1, Area 2",
			"energy spectrum : behind the top right ball (Aluminium)",
			BIN, EMIN, EMAX );
	TH1F* energyRun1Area3 = new TH1F( "Energy RunID = 1, Area 3",
			"energy spectrum : behind the bottom left ball (Glass)",
			BIN, EMIN, EMAX );
	TH1F* energyRun1Area4 = new TH1F( "Energy RunID = 1, Area 4",
			"energy spectrum : behind the bottom right ball (SpineBone)",
			BIN, EMIN, EMAX );

	// Count the number of pixels in a set
	// Run 0
	// 0utside of the phantom
	set<Double_t> pixelNoPhantom;
	Int_t countRun0NoPhantom = 0;
	// Cylinder water
	set<Double_t> pixelCylinder;
	Int_t countRun0Cylinder = 0;
	// 2 top balls
	set<Double_t> pixelBalls;
	Int_t countRun0TopBalls = 0;
	// 2 bottom Balls
	Int_t countRun0BottomBalls = 0;

	// Run 1
	// Outside of the phantom
	Int_t countRun1NoPhantom = 0;
	// Cylinder water
	Int_t countRun1Cylinder = 0;
	// In the top left ball
	set<Double_t> pixelBallsRun1;
	Int_t countRun1TopLeftBalls = 0;
	// In the top right ball
	Int_t countRun1TopRightBalls = 0;
	// In the bottom left ball
	Int_t countRun1BottomLeftBalls = 0;
	// In the bottom right ball
	Int_t countRun1BottomRightBalls = 0;

	for( Int_t i = 0; i != entriesSingleTree; ++i )
	{
		singlesTree->GetEntry( i );
		if( runID == 0 )
		{
			run_0->Fill( globalPosX, globalPosY );
			if( globalPosX < -16 || globalPosX > 16 )
			{
				energyRun0Area1->Fill( energy );
				pixelNoPhantom.insert( pixelID );
				++countRun0NoPhantom;
			}
			else if( ( globalPosX > -3.5 && globalPosX < 3.5 )
					&& ( globalPosY < 19 && globalPosY > -19 ) )
			{
				energyRun0Area2->Fill( energy );
				pixelCylinder.insert( pixelID );
				++countRun0Cylinder;
			}
			else if( ( globalPosX > 8 && globalPosX < 11 )
					&& ( globalPosY < 14 && globalPosY > 10.5 ) )
			{
				energyRun0Area3->Fill( energy );
				pixelBalls.insert( pixelID );
				++countRun0TopBalls;
			}
			else if( ( globalPosX > -11 && globalPosX < -8 )
					&& ( globalPosY < -10.5 && globalPosY > -14 ) )
			{
				energyRun0Area4->Fill( energy );
				++countRun0BottomBalls;
			}
		}
		if( runID == 1 )
		{
			run_1->Fill( globalPosX, globalPosY );
			if( ( globalPosX > 4 && globalPosX < 7 )
					&& ( globalPosY > 10 && globalPosY < 13.5 ) )
			{
				energyRun1Area2->Fill( energy );
				pixelBallsRun1.insert( pixelID );
				++countRun1TopRightBalls;
			}
			else if( ( globalPosX > 4 && globalPosX < 7 )
					&& ( globalPosY > -14.5 && globalPosY < -11 ) )
			{
				energyRun1Area4->Fill( energy );
				++countRun1TopLeftBalls;
			}
			else if( ( globalPosX > -7 && globalPosX < -4 )
					&& ( globalPosY > 10 && globalPosY < 13.5 ) )
			{
				energyRun1Area1->Fill( energy );
				++countRun1BottomRightBalls;
			}
			else if( ( globalPosX > -7 && globalPosX < -4 )
					&& ( globalPosY > -14.5 && globalPosY < -11 ) )
			{
				energyRun1Area3->Fill( energy );
				++countRun1BottomLeftBalls;
			}
			else if( globalPosX < -16 || globalPosX > 16 )
				++countRun1NoPhantom;
			else if( ( globalPosX > -3.5 && globalPosX < 3.5 )
					&& ( globalPosY < 19 && globalPosY > -19 ) )
				++countRun1Cylinder;
		}
	}

	Double_t nbrPixelAreaNoPhantom = pixelNoPhantom.size();
	Double_t nbrPixelAreaCylinder = pixelCylinder.size();
	Double_t nbrPixelAreaBallsRun0 = pixelBalls.size();
	Double_t nbrPixelAreaBallsRun1 = pixelBallsRun1.size();

	// Statistics
	// Deviation standard for each run, in 2 regions of interest
	// - Without the phantom
	// - In the water cylinder
	// *** The first run
	cout << "*****************************" << endl;
	cout << "* First Run (theta = 0) =>  *" << endl;
	cout << "*****************************" << endl;
	cout << "- Outside of the phantom :" << endl;
	cout << "--------------------------" << endl;
	cout << "Mean : "
		 << countRun0NoPhantom / nbrPixelAreaNoPhantom
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun0NoPhantom / nbrPixelAreaNoPhantom ) << endl;
	cout << endl;
	cout << "- In the water cylinder :" << endl;
	cout << "-------------------------" << endl;
	cout << "Mean : "
		 << countRun0Cylinder / nbrPixelAreaCylinder
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun0Cylinder / nbrPixelAreaCylinder ) << endl;
	cout << endl;
	cout << "- Behind the 2 top balls : " << endl;
	cout << "--------------------------" << endl;
	cout << "Mean : "
		 << countRun0TopBalls / nbrPixelAreaBallsRun0
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun0TopBalls / nbrPixelAreaBallsRun0 ) << endl;
	cout << endl;
	cout << "- Behind the 2 bottom balls : " << endl;
	cout << "-----------------------------" << endl;
	cout << "Mean : "
		 << countRun0BottomBalls / nbrPixelAreaBallsRun0
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun0BottomBalls / nbrPixelAreaBallsRun0 ) << endl;

	// **** the second run
	cout << endl;
	cout << "*******************************" << endl;
	cout << "* Second Run (theta = 90) =>  *" << endl;
	cout << "*******************************" << endl;
	cout << "- Outside of the phantom :" << endl;
	cout << "--------------------------" << endl;
	cout << "Mean : "
		 << countRun1NoPhantom / nbrPixelAreaNoPhantom
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun1NoPhantom / nbrPixelAreaNoPhantom ) << endl;
	cout << endl;
	cout << "- In the water cylinder :" << endl;
	cout << "-------------------------" << endl;
	cout << "Mean : "
		 << countRun1Cylinder / nbrPixelAreaCylinder
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun1Cylinder / nbrPixelAreaCylinder ) << endl;
	cout << endl;
	cout << "- Behind the top left ball (PVC) :" << endl;
	cout << "----------------------------------" << endl;
	cout << "Mean : "
		 << countRun1TopLeftBalls / nbrPixelAreaBallsRun1
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun1TopLeftBalls / nbrPixelAreaBallsRun1 ) << endl;
	cout << endl;
	cout << "- Behind the top right ball (Aluminium) :" << endl;
	cout << "-----------------------------------------" << endl;
	cout << "Mean : "
		 << countRun1TopRightBalls / nbrPixelAreaBallsRun1
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun1TopRightBalls / nbrPixelAreaBallsRun1 ) << endl;
	cout << endl;
	cout << "- Behind the bottom left ball (Glass) :" << endl;
	cout << "---------------------------------------" << endl;
	cout << "Mean : "
		 << countRun1BottomLeftBalls / nbrPixelAreaBallsRun1
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun1BottomLeftBalls / nbrPixelAreaBallsRun1 ) << endl;
	cout << endl;
	cout << "- Behind the bottom right ball (SpineBone) :" << endl;
	cout << "--------------------------------------------" << endl;
	cout << "Mean : "
		 << countRun1BottomRightBalls / nbrPixelAreaBallsRun1
		 << " ph./pix." << endl;
	cout << "Standard Deviation : "
		 << sqrt( countRun1BottomRightBalls / nbrPixelAreaBallsRun1 ) << endl;

	gStyle->SetPalette( 1 );
	gStyle->SetOptStat( "ne" );

	// First Pad : the projection of run_0
	TPad* padRun_0 = new TPad( "padRun_0", "first projection",
			0.125, 0.68, 0.375, 0.98 );

	// Second Pad : the projection of run_1
	TPad* padRun_1 = new TPad( "padRun_1", "second projection",
			0.625, 0.68, 0.875, 0.98 );

	// Third pad : energy spectrum without phantom
	TPad* padEnergy_0_Run_0 = new TPad( "padEnergy_0_Run_0",
			"first energy spectrum", 0.01, 0.35, 0.24, 0.65 );

	// Fourth pad : energy spectrum in the water cylinder
	TPad* padEnergy_1_Run_0 = new TPad( "padEnergy_1_Run_0",
			"second energy spectrum", 0.26, 0.35, 0.49, 0.65 );

	// Fifth pad : energy spectrum in the top balls
	TPad* padEnergy_2_Run_0 = new TPad( "padEnergy_2_Run_0",
			"third energy spectrum", 0.01, 0.02, 0.24, 0.32 );

	// Sixth pad : energy spectrum in the bottom balls
	TPad* padEnergy_3_Run_0 = new TPad( "padEnergy_3_Run_0",
			"fourth energy spectrum", 0.26, 0.02, 0.49, 0.32 );

	// Third pad : energy spectrum in the top left ball (PVC)
	TPad* padEnergy_0_Run_1 = new TPad( "padEnergy_0_Run_1",
			"fifth energy spectrum", 0.51, 0.35, 0.74, 0.65 );

	// Fourth pad : energy spectrum in the top right ball (Aluminium)
	TPad* padEnergy_1_Run_1 = new TPad( "padEnergy_1_Run_1",
			"sixth energy spectrum", 0.76, 0.35, 0.99, 0.65 );

	// Fifth pad : energy spectrum in the bottom left ball (Glass)
	TPad* padEnergy_2_Run_1 = new TPad( "padEnergy_2_Run_1",
			"seventh energy spectrum", 0.51, 0.02, 0.74, 0.32 );

	// Sixth pad : energy spectrum in the bottom right ball (SpineBone)
	TPad* padEnergy_3_Run_1 = new TPad( "padEnergy_3_Run_1",
			"eighth energy spectrum", 0.76, 0.02, 0.99, 0.32 );

	// Draw the Pads
	// Run 0
	padRun_0->SetFillStyle( 4000 );
	padRun_0->Draw();
	// Energy
	padEnergy_0_Run_0->SetFillStyle( 4000 );
	padEnergy_1_Run_0->SetFillStyle( 4000 );
	padEnergy_2_Run_0->SetFillStyle( 4000 );
	padEnergy_3_Run_0->SetFillStyle( 4000 );
	padEnergy_0_Run_0->Draw();
	padEnergy_1_Run_0->Draw();
	padEnergy_2_Run_0->Draw();
	padEnergy_3_Run_0->Draw();

	// Run 1
	padRun_1->SetFillStyle( 4000 );
	padRun_1->Draw();
	// Energy
	padEnergy_0_Run_1->SetFillStyle( 4000 );
	padEnergy_1_Run_1->SetFillStyle( 4000 );
	padEnergy_2_Run_1->SetFillStyle( 4000 );
	padEnergy_3_Run_1->SetFillStyle( 4000 );
	padEnergy_0_Run_1->Draw();
	padEnergy_1_Run_1->Draw();
	padEnergy_2_Run_1->Draw();
	padEnergy_3_Run_1->Draw();

	// In padRun_0
	padRun_0->cd();
	run_0->Draw( "COLZ" );

	// In padRun_1
	padRun_1->cd();
	run_1->Draw( "COLZ" );

	// In padEnergy_0_Run_0
	padEnergy_0_Run_0->cd();
	padEnergy_0_Run_0->SetLogy();
	energyRun0Area1->SetMaximum( 5000000 );
	energyRun0Area1->Draw();
	energyRun0Area1->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_1_Run_0
	padEnergy_1_Run_0->cd();
	padEnergy_1_Run_0->SetLogy();
	energyRun0Area2->SetMaximum( 500000 );
	energyRun0Area2->Draw();
	energyRun0Area2->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_2_Run_0
	padEnergy_2_Run_0->cd();
	padEnergy_2_Run_0->SetLogy();
	energyRun0Area3->SetMaximum( 50000 );
	energyRun0Area3->Draw();
	energyRun0Area3->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_3_Run_0
	padEnergy_3_Run_0->cd();
	padEnergy_3_Run_0->SetLogy();
	energyRun0Area4->SetMaximum( 50000 );
	energyRun0Area4->Draw();
	energyRun0Area4->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_0_Run_1
	padEnergy_0_Run_1->cd();
	padEnergy_0_Run_1->SetLogy();
	energyRun1Area1->SetMaximum( 50000 );
	energyRun1Area1->Draw();
	energyRun1Area1->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_1_Run_1
	padEnergy_1_Run_1->cd();
	padEnergy_1_Run_1->SetLogy();
	energyRun1Area2->SetMaximum( 50000 );
	energyRun1Area2->Draw();
	energyRun1Area2->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_2_Run_1
	padEnergy_2_Run_1->cd();
	padEnergy_2_Run_1->SetLogy();
	energyRun1Area3->SetMaximum( 50000 );
	energyRun1Area3->Draw();
	energyRun1Area3->GetXaxis()->SetTitle( "Energy (in MeV)" );

	// In padEnergy_3_Run_1
	padEnergy_3_Run_1->cd();
	padEnergy_3_Run_1->SetLogy();
	energyRun1Area4->SetMaximum( 50000 );
	energyRun1Area4->Draw();
	energyRun1Area4->GetXaxis()->SetTitle( "Energy (in MeV)" );

	delete singlesTree;
	delete gateTree;

	app.Run();

	return 0;
}
