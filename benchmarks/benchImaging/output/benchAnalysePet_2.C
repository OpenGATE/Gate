{
/////////////////////////////////////////////////////   
//                                                 //
//         S.JAN - sebastien.jan@cea.fr - March 2014     //
/////////////////////////////////////////////////////

// READ the output File
   TFile *f = new TFile("pet_results.root");

// Connect the TTree defined in the output file
   TTree *Coincidences = (TTree*)gDirectory->Get("Coincidences");
   TTree *Singles = (TTree*)gDirectory->Get("Singles");

// Statistical informations 
   gStyle -> SetStatW(0.28);
   gStyle -> SetStatH(0.13);
   gStyle -> SetStatColor(41);   
   gStyle -> SetStatX(0.87);
   gStyle -> SetStatY(0.85);   
   gStyle -> SetStatFont(42);
   gStyle->SetOptStat(111);

   
 
// Histo 2D creation 
TH2F *fantome = new TH2F("fantome","",100,-100,100,100,-100,100);  

 
//
// Declaration of leaves types - TTree Coincidences
//  
  
   Float_t         sourcePosX;
   Float_t         sourcePosY;
   Float_t         sourcePosZ;
   
   
//   
// Set branch addresses - TTree Coincicences
//  
  
   Singles->SetBranchAddress("sourcePosX",&sourcePosX);
   Singles->SetBranchAddress("sourcePosY",&sourcePosY);
   Singles->SetBranchAddress("sourcePosZ",&sourcePosZ);
   
   

   Int_t nentries = Singles->GetEntries();
   Int_t nbytes = 0;
    
//
// Loop on the events in the TTree Coincidences
//


    for (Int_t i=0; i<nentries;i++) {
       nbytes += Singles->GetEntry(i);
       
       fantome->Fill(sourcePosX,sourcePosY);
       
                                    }
				    
	    
//
// Plot the results
//

   gStyle->SetPalette(1);
   
   
   c1 = new TCanvas("c1","Reco",200,10,500,600);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);   
       
   fantome.Draw("contZ");
   
   c1->Update(); 


}
