{

//
// Initialisation
//
   gROOT->Reset();
   TFile f("root.root");
   TTree *Singles = (TTree*)gDirectory->Get("Singles");

//
// Declaration of leaves types - TTree Singles
//  
   Int_t           RayleighCrystal;
   Int_t           RayleighPhantom;
   Int_t           comptonCrystal;
   Int_t           comptonPhantom;
   Float_t         energy;
   Float_t         globalPosX;
   Float_t         globalPosY;
   Float_t         globalPosZ;
   Float_t         sourcePosX;
   Float_t         sourcePosY;
   Float_t         sourcePosZ;

//   
// Set branch addresses - TTree Singles
//  
   Singles->SetBranchAddress("RayleighCrystal",&RayleighCrystal);
   Singles->SetBranchAddress("RayleighPhantom",&RayleighPhantom);
   Singles->SetBranchAddress("comptonCrystal",&comptonCrystal);
   Singles->SetBranchAddress("comptonPhantom",&comptonPhantom);
   Singles->SetBranchAddress("energy",&energy);
   Singles->SetBranchAddress("globalPosX",&globalPosX);
   Singles->SetBranchAddress("globalPosY",&globalPosY);
   Singles->SetBranchAddress("globalPosZ",&globalPosZ);
   Singles->SetBranchAddress("sourcePosX",&sourcePosX);
   Singles->SetBranchAddress("sourcePosY",&sourcePosY);
   Singles->SetBranchAddress("sourcePosZ",&sourcePosZ);

   /////////////STAT////////   
   gStyle -> SetStatW(0.28);
   gStyle -> SetStatH(0.3);
   gStyle -> SetStatColor(41);   
   gStyle -> SetStatX(1);
   gStyle -> SetStatY(1);   
   gStyle -> SetStatFont(42);
   gStyle->SetOptStat(0);
   gStyle->SetOptFit(0);
   /////////////////////////

   TH1F *EnergyTrue     = new TH1F("EnergyTrue","Energy distribution of unscattered Singles",120,0.,+700.);
   TH1F *EnergyScatter  = new TH1F("EnergyScatter","Energy distribution of scattered Singles",120,0.,+700.);

   TH1F *DetectPosAxial = new TH1F("DetectPosAxial","Axial detection distribution",50,-200.,+200.);
   
   TH2F *DetectPos      = new TH2F("DetectPos","Transaxial detection position",252,-504.,+504.,252,-504.,+504.);

   TH3F *Detect3D       = new TH3F("Detect3D","Global phantom view", 100,-200,+200,100,-200,+200,100,-400,+400);

   Int_t nentries = Singles->GetEntries();
   Int_t nbytes = 0, nscatter = 0, ntrue = 0, nscatCrystal = 0, ncompton = 0, nrayleigh = 0, nboth = 0;
   
   Double_t StartTime   = 0.;
   Double_t StopTime    = 1.;

//
// Loop for each event in the TTree Singles
//
   for (Int_t i=0; i<nentries;i++) {
     nbytes += Singles->GetEntry(i);

     DetectPosAxial->Fill(globalPosZ);
     DetectPos->Fill(globalPosX,globalPosY);
     DetectPos->Fill(sourcePosX,sourcePosY);
     Detect3D->Fill(globalPosX,globalPosY,globalPosZ);
     Detect3D->Fill(sourcePosX,sourcePosY,sourcePosZ);

     // True unscattered single
     if (comptonPhantom == 0 && RayleighPhantom == 0) {
	 ntrue++;
         EnergyTrue->Fill(energy*1000.);
     }
     // True scattered single
     else {
         if (comptonPhantom != 0 && RayleighPhantom != 0) nboth++;
         else if (comptonPhantom != 0) ncompton++; // compton scattering
         else nrayleigh++; // Rayleigh scattering
         nscatter++; // Both compton and Rayleigh scattering
         EnergyScatter->Fill(energy*1000.);
     }
     // Crystal scattering
     if (comptonCrystal != 0 || RayleighCrystal != 0) {
         nscatCrystal++;
     } 
   }

//
// Print out results
//
   cout << endl << endl;
   cout << " ************************************************************** " << endl;
   cout << " *                                                            * " << endl;
   cout << " *   B E N C H M A R K    R E S U L T S    A N A L Y S I S    * " << endl;
   cout << " *                                                            * " << endl;
   cout << " ************************************************************** " << endl;
   cout << " *" << endl;
   cout << " *                 --> 25*25*25 matrix" << endl;
   cout << " *                 --> 3 materials" << endl;
   cout << " *                 --> positrons" << endl;
   cout << " *" << endl << " *" << endl;     

   cout << " *  Total number of Singles : " << nentries << endl;
   cout << " *" << endl;
   cout << " *  Total number of true    : " << ntrue << endl;
   cout << " *    --> percentage : " << ((float)ntrue)*100./((float)nentries) << endl;
   cout << " *  Total number of scatter : " << nscatter << endl;
   cout << " *    --> percentage : " << ((float)nscatter)*100./((float)nentries) << endl;
   cout << " *" << endl;

   cout << " *  Number of compton  : " << ncompton << endl;
   cout << " *    --> percentage : " << ((float)ncompton)*100./((float)nscatter) << endl;
   cout << " *  Number of Rayleigh : " << nrayleigh << endl;
   cout << " *    --> percentage : " << ((float)nrayleigh)*100./((float)nscatter) << endl;
   cout << " *  Number of both     : " << nboth << endl;
   cout << " *    --> percentage : " << ((float)nboth)*100./((float)nscatter) << endl;
   cout << " *" << endl;

   cout << " *  Total number of scatter in crystal : " << nscatCrystal << endl;
   cout << " *    --> percentage : " << ((float)nscatCrystal)*100./((float)nentries) << endl;
   cout << " *" << endl;
   cout << " ************************************************************** " << endl;
   cout << endl;
 
   c1 = new TCanvas("c1","Bench",0,0,1000,700);
   Int_t pos=1;
   c1->Divide(3,2);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->cd(pos++);
   DetectPosAxial->Draw();
   c1->cd(pos++);
   DetectPos->Draw();
   c1->cd(pos++);
   Detect3D->Draw();  
   c1->cd(pos++);
   EnergyTrue->Draw();
   c1->cd(pos++);
   EnergyScatter->Draw();
   c1->Update();   
   c1->SaveAs("bench.gif");
}	
