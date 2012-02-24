/////////////////////////////////////////////////////   
//                                                 //
//         S.JAN - sjan@cea.fr - March 2007        //
//			updated by							   //
//			U. Pietrzyk - u.pietrzyk@fz-juelich.de //
//			March 2010							   //
//                                                 //
//   Example of a ROOT C++ code-macro to:          // 
//   -----------------------------------           //
//   1/ read an output root data file              //
//   2/ create a loop on each event which are      //
//      stored during the simulation               //
//   3/ perform data processing                    //
//   4/ plot the results                           //
//                                                 //
/////////////////////////////////////////////////////

void VoxelPhantom_Analyse()
{
	
//Reset ROOT and connect tree file
	
   gROOT->Reset();
   TFile *f = new TFile("data_Voxel_Phantom.root");
   
   TTree *Singles = (TTree*)gDirectory->Get("Singles");
   
/////////////STAT////////   
   gStyle -> SetStatW(0.28);
   gStyle -> SetStatH(0.13);
   gStyle -> SetStatColor(41);   
   gStyle -> SetStatX(0.87);
   gStyle -> SetStatY(0.85);   
   gStyle -> SetStatFont(42);
   gStyle->SetOptStat(111);
/////////////////////////
   
 
//creation histo 2 Dim. 
TH2F *Phantom = new TH2F("Phantom","",100,-100,100,100,-100,100);  

 
//
//Declaration of leaves types - TTree Singles
//  
  
   Float_t         sourcePosX;
   Float_t         sourcePosY;
   Float_t         sourcePosZ;
   
   
//   
//Set branch addresses - TTree Singles
//  
  
   Singles->SetBranchAddress("sourcePosX",&sourcePosX);
   Singles->SetBranchAddress("sourcePosY",&sourcePosY);
   Singles->SetBranchAddress("sourcePosZ",&sourcePosZ);
   
   

   Int_t nentries = Singles->GetEntries();
   Int_t nbytes = 0;
    
//
//Loop on event number for Singles TTree
//


    for (Int_t i=0; i<nentries;i++) {
       nbytes += Singles->GetEntry(i);
       
       Phantom->Fill(sourcePosX,sourcePosY);
       
	}
				    
	    
//
// Result plots
//

   gStyle->SetPalette(1);
   
   
   c1 = new TCanvas("Reco","Reco",200,10,500,600);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);   
       
   Phantom.Draw("contZ");
   
   c1->Update(); 


}
