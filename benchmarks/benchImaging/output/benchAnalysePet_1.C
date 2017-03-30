/////////////////////////////////////////////////////   
//                                                 //
//         S.JAN - sebastien.jan@cea.fr - March 2014     //
/////////////////////////////////////////////////////
{

// READ the output File
   TFile *f = new TFile("pet_results.root");

// Connect the TTree defined in the output file
   TTree *Coincidences = (TTree*)gDirectory->Get("Coincidences");

   //TTree *Gate = (TTree*)gDirectory->Get("Gate");
   //TTree *Hits = (TTree*)gDirectory->Get("Hits");
   //TTree *Singles = (TTree*)gDirectory->Get("Singles");
   

   
// Histo 1D creation
TH1F *gamma1 = new TH1F("gamma1","",80,0.2,.8);  
TH1F *gamma2 = new TH1F("gamma2","",100,0.2,.8);  

 
// Histo 2D creation 
TH3F *position = new TH3F("position","",200,-400,400,200,-400,400,200,-400,400);  

  
//
// Declaration of leaves types - TTree Coincidences
//  
   Float_t         axialPos;
   Char_t          comptVolName1[40];
   Char_t          comptVolName2[40];
   Int_t           comptonPhantom1;
   Int_t           comptonPhantom2;
   Int_t           comptonCrystal1;
   Int_t           comptonCrystal2;   
   Int_t           crystalID1;
   Int_t           crystalID2;
   //Int_t           blockID1;
   //Int_t           blockID2;
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
   Double_t         time1;
   Double_t         time2;
   
//   
// Set branch addresses - TTree Coincicences
//  
   Coincidences->SetBranchAddress("axialPos",&axialPos);
   Coincidences->SetBranchAddress("comptVolName1",&comptVolName1);
   Coincidences->SetBranchAddress("comptVolName2",&comptVolName2);
   Coincidences->SetBranchAddress("comptonPhantom1",&comptonPhantom1);
   Coincidences->SetBranchAddress("comptonPhantom2",&comptonPhantom2);
   Coincidences->SetBranchAddress("comptonCrystal1",&comptonCrystal1);
   Coincidences->SetBranchAddress("comptonCrystal2",&comptonCrystal2);
   Coincidences->SetBranchAddress("crystalID1",&crystalID1);
   Coincidences->SetBranchAddress("crystalID2",&crystalID2);
  // Coincidences->SetBranchAddress("blockID1",&blockID1);
  // Coincidences->SetBranchAddress("blockID2",&blockID2);
   Coincidences->SetBranchAddress("energy1",&energy1);
   Coincidences->SetBranchAddress("energy2",&energy2);   
   Coincidences->SetBranchAddress("eventID1",&eventID1);
   Coincidences->SetBranchAddress("eventID2",&eventID2);
   Coincidences->SetBranchAddress("globalPosX1",&globalPosX1);
   Coincidences->SetBranchAddress("globalPosX2",&globalPosX2);
   Coincidences->SetBranchAddress("globalPosY1",&globalPosY1);
   Coincidences->SetBranchAddress("globalPosY2",&globalPosY2);      
   Coincidences->SetBranchAddress("globalPosZ1",&globalPosZ1);
   Coincidences->SetBranchAddress("globalPosZ2",&globalPosZ2);
   Coincidences->SetBranchAddress("layerID1",&layerID1);
   Coincidences->SetBranchAddress("layerID2",&layerID2);
   Coincidences->SetBranchAddress("moduleID1",&moduleID1);
   Coincidences->SetBranchAddress("moduleID2",&moduleID2);
   Coincidences->SetBranchAddress("rotationAngle",&rotationAngle);
   Coincidences->SetBranchAddress("rsectorID1",&rsectorID1);
   Coincidences->SetBranchAddress("rsectorID2",&rsectorID2);
   Coincidences->SetBranchAddress("runID",&runID);
   Coincidences->SetBranchAddress("sinogramS",&sinogramS);
   Coincidences->SetBranchAddress("sinogramTheta",&sinogramTheta);
   Coincidences->SetBranchAddress("sourceID1",&sourceID1);
   Coincidences->SetBranchAddress("sourceID2",&sourceID2);
   Coincidences->SetBranchAddress("sourcePosX1",&sourcePosX1);
   Coincidences->SetBranchAddress("sourcePosX2",&sourcePosX2);
   Coincidences->SetBranchAddress("sourcePosY1",&sourcePosY1);
   Coincidences->SetBranchAddress("sourcePosY2",&sourcePosY2);
   Coincidences->SetBranchAddress("sourcePosZ1",&sourcePosZ1);
   Coincidences->SetBranchAddress("sourcePosZ2",&sourcePosZ2);
   Coincidences->SetBranchAddress("submoduleID1",&submoduleID1);
   Coincidences->SetBranchAddress("submoduleID2",&submoduleID2);
   Coincidences->SetBranchAddress("time1",&time1);
   Coincidences->SetBranchAddress("time2",&time2);
   
   

   Int_t nentries = Coincidences->GetEntries();
   Int_t nbytes = 0;
    
//
// Loop on the events in the TTree Coincidences
//

Float_t Nbr_Coinc_Prompt = 0. ;
Float_t Nbr_Coinc_Random = 0. ;
Float_t Nbr_Coinc_Scatter = 0. ;
Float_t Nbr_Coinc_Trues = 0. ;
Float_t Ntot = 10000000. ;

    for (Int_t i=0; i<nentries;i++) {
       nbytes += Coincidences->GetEntry(i);
       
       
       // Fill gamma1 histo without condition
       gamma1->Fill(energy1);
       
       // Fill the gamma2 histo with condition
       if (energy2 >= 0.4)gamma2->Fill(energy2);
       
       // Fill the 3D Histo without condition
       position->Fill(globalPosZ1,globalPosX1,globalPosY1);
       
       
       ///////////////////////////////////////////////////////////////////////////////////////////
       //      						                                        //	
       // E V A L U A T I O N   O F   :   P R O M P T S   ;   T R U E S   ;   R A N D O M   ;   //
       //                                                                                       //
       // S E N S I T I V I T Y                                                                 //                           
       //									                //
       ///////////////////////////////////////////////////////////////////////////////////////////
		
		Nbr_Coinc_Prompt++;
		
		if ( eventID1 != eventID2 ) Nbr_Coinc_Random++;
            
                if ( eventID1 == eventID2 && comptonPhantom1 == 0 && comptonPhantom2 == 0 ) Nbr_Coinc_Trues++;
			    
				    
				    
				    }

Float_t Sensi = Nbr_Coinc_Prompt/Ntot*100.;


cout<<""<<endl;
cout<<""<<endl;
cout<<""<<endl;
cout<<"#   P R O M P T S     =   "<<Nbr_Coinc_Prompt <<"   Cps"<<endl;
cout<<"#   T R U E S         =   "<<Nbr_Coinc_Trues  <<"   Cps"<<endl;
cout<<"#   R A N D O M S     =   "<<Nbr_Coinc_Random <<"   Cps"<<endl;
cout<<" ______________________________________"<<endl;
cout<<"|                                      "<<endl;
cout<<"|  T O T A L   S E N S I T I V I T Y   :   "<<  Sensi <<"  %"<<endl;
cout<<"|______________________________________"<<endl;
cout<<""<<endl;
cout<<""<<endl;
cout<<""<<endl; 


//
// Plot the results
//

   gStyle->SetPalette(1);
   
   
   c1 = new TCanvas("c1","Reco",200,10,500,600);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);   
       
   gamma1->Draw();
   gamma1->SetFillColor(2);
   gamma2->Draw("same");  
   gamma2->SetFillColor(9);
   
   
   tex = new TLatex(0.255919,4523.54,"GAMMA 1");
   tex->SetTextColor(2);
   tex->SetTextSize(0.05);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.620151,2778.42,"GAMMA 2");
   tex->SetTextColor(9);
   tex->SetTextSize(0.05);
   tex->SetLineWidth(2);
   tex->Draw();
     
   c1->Update(); 

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
   c2 = new TCanvas("c2","Reco_true",200,10,500,600);
   c2->SetFillColor(0);
   c2->SetGrid();
   c2->SetBorderMode(0); 

   position->Draw();
   
   c2->Update(); 
    
}
