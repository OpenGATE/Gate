{
   gROOT->Reset();
   TFile f("benchmarkPET.root");

   TTree *Coincidences = (TTree*)gDirectory->Get("Coincidences");
   TTree *Hits = (TTree*)gDirectory->Get("Hits");
   TTree *Singles = (TTree*)gDirectory->Get("Singles");
   TTree *delay = (TTree*)gDirectory->Get("delay");

//
//Declaration of leaves types - TTree Coincidences
//  
   Int_t           RayleighCrystal1;
   Int_t           RayleighCrystal2;
   Int_t           RayleighPhantom1;
   Int_t           RayleighPhantom2;
   Char_t          RayleighVolName1[40];
   Char_t          RayleighVolName2[40];
   Float_t         axialPos;
   Char_t          comptVolName1[40];
   Char_t          comptVolName2[40];
   Int_t           compton1;
   Int_t           compton2;
   Int_t           crystalID1;
   Int_t           crystalID2;
   Int_t           comptonPhantom1;
   Int_t           comptonPhantom2;
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
   
   Float_t         zmin,zmax,z;
//   
//Set branch addresses - TTree Coincicences
//  
   Coincidences->SetBranchAddress("RayleighCrystal1",&RayleighCrystal1);
   Coincidences->SetBranchAddress("RayleighCrystal2",&RayleighCrystal2);
   Coincidences->SetBranchAddress("RayleighPhantom1",&RayleighPhantom1);
   Coincidences->SetBranchAddress("RayleighPhantom2",&RayleighPhantom2);
   Coincidences->SetBranchAddress("RayleighVolName1",&RayleighVolName1);
   Coincidences->SetBranchAddress("RayleighVolName2",&RayleighVolName2);
   Coincidences->SetBranchAddress("axialPos",&axialPos);
   Coincidences->SetBranchAddress("comptVolName1",&comptVolName1);
   Coincidences->SetBranchAddress("comptVolName2",&comptVolName2);
   Coincidences->SetBranchAddress("comptonCrystal1",&compton1);
   Coincidences->SetBranchAddress("comptonCrystal2",&compton2);
   Coincidences->SetBranchAddress("crystalID1",&crystalID1);
   Coincidences->SetBranchAddress("crystalID2",&crystalID2);
   Coincidences->SetBranchAddress("comptonPhantom1",&comptonPhantom1);
   Coincidences->SetBranchAddress("comptonPhantom2",&comptonPhantom2);
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

   TH1F *DecayO15 = new TH1F("Decroissance 0-15","O-15 decay",60,0.,240.);
   TH1F *DecayF18 = new TH1F("Decroissance F-18","F-18 decay",60,0.,240.);
   TH1F *DetectPosAxial = new TH1F("DetectPosAxial","Axial detection position",52.,-208.,+208.);

   TH1F *AxialSensitivityDet = new TH1F("AxialSensitivityDet","Axial sensitivity",50,-200.,+200.);
   TH1F *AxialScattersDet = new TH1F("AxialScattersDet","Axial scatters distribution",50,-200.,+200.);
   TH1F *AxialProfileDet = new TH1F("AxialProfileDet","",50,-200.,+200.);
   TH1F *ScatterFractionAxialDet = new TH1F("ScatterFractionAxialDet","Axial scatter fraction",50,-200.,+200.);
   
   TH2F *DetectPos1 = new TH2F("DetectPos1","Transaxial detection position",252,-504.,+504.,252,-504.,+504.);
   TH2F *DetectPos2 = new TH2F("DetectPos2","Transaxial detection position",252,-504.,+504.,252,-504.,+504.);
   Int_t nentries = Coincidences->GetEntries();
   Int_t nbytes = 0, nbytesdelay = 0, nrandom = 0, nscatter = 0, ntrue = 0;
   
   Double_t O15Activity = 100000.;
   Double_t F18Activity = 100000.;
   Double_t StartTime   = 0.;
   Double_t StopTime    = 240.;
   Double_t O15HalfLife = 122.24;
   Double_t F18HalfLife = 6586.2;    
   Double_t O15DecayFactor = (1.0 - exp(-log(2.0)*(StopTime-StartTime)/O15HalfLife))/
                             (exp(+log(2.0)*StartTime/O15HalfLife)*log(2.0)/O15HalfLife*(StopTime-StartTime));    
   Double_t F18DecayFactor = (1.0 - exp(-log(2.0)*(StopTime-StartTime)/F18HalfLife))/
                             (exp(+log(2.0)*StartTime/F18HalfLife)*log(2.0)/F18HalfLife*(StopTime-StartTime)); 
   Double_t O15Decay = O15Activity * (StopTime-StartTime) * O15DecayFactor;			     
   Double_t F18Decay = F18Activity * (StopTime-StartTime) * F18DecayFactor;			     


//
// Loop for each event in the TTree Coincidences
//
   for (Int_t i=0; i<nentries;i++) {
     if (fmod((double)i,10000.0) == 0.0) cout << ".";
     nbytes += Coincidences->GetEntry(i);
     if (eventID1 != eventID2) { // Random coincidence
       ++nrandom;
     } else {  // True coincidence
       if (runID == 0) { // First frame
         DetectPos1->Fill(globalPosX1,globalPosY1);
         DetectPos1->Fill(globalPosX2,globalPosY2);
         DetectPos1->Fill(sourcePosX1,sourcePosY1);
       }  else if (runID == 1) { // Second frame
         DetectPos2->Fill(globalPosX1,globalPosY1);
         DetectPos2->Fill(globalPosX2,globalPosY2);
         DetectPos2->Fill(sourcePosX1,sourcePosY1);
       }
       DetectPosAxial->Fill(globalPosZ1);
       DetectPosAxial->Fill(globalPosZ2);
       z = (globalPosZ1+globalPosZ2)/2.;
       if (comptonPhantom1 == 0 && comptonPhantom2 == 0 &&
           RayleighPhantom1 == 0 && RayleighPhantom2 == 0) {  // true unscattered coincidence
         AxialSensitivityDet->Fill(z);
	 ntrue++;
       } else { // true scattered coincidence
         AxialScattersDet->Fill(z);
         nscatter++;
       }  
       AxialProfileDet->Fill(z);
       if ((sourceID1 == 1) && (sourceID2 == 1)) DecayO15->Fill(time1);
       else if ((sourceID1 == 0) && (sourceID2 == 0)) DecayF18->Fill(time1);
     }  
   }
   cout << endl << endl;
   e1 = new TF1("e1","expo",0.,240.);
   //g1 = new TF1("g1","gaus",-5.,+5.);
   //Acolinea_Angle_Distribution_deg->Fit("g1");
   DecayO15->Fit("e1","","",0.,240.);
   //Double_t ndecay = Ion_decay_time_s->GetEntries();
   cout << endl << endl;     
   cout << " *********************************************************************************** " << endl;
   cout << " *                                                                                 * " << endl;
   cout << " *   G A T E    P E T    B E N C H M A R K    R E S U L T S    A N A L Y S I S     * " << endl;
   cout << " *                                                                                 * " << endl;
   cout << " *********************************************************************************** " << endl;
   cout << endl << endl;     
   cout << " Acquisition start time = " << StartTime << " [s]" << endl;
   cout << " Acquisition stop time  = " << StopTime  << " [s]" << endl;
   cout << " O-15 decay factor = " << O15DecayFactor << endl;
   cout << " F-18 decay factor = " << F18DecayFactor << endl;
   cout << " O-15 initial activity = " << O15Activity << " [Bq]" << endl;
   cout << " F-18 initial activity = " << F18Activity << " [Bq]" << endl;   
   cout << " O-15 decays = " << O15Decay << endl;			        
   cout << " F-18 decays = " << F18Decay << endl;			        
   cout << " ==> Expected total number of decays during the acquisition is " << O15Decay+F18Decay << " +/- " << sqrt(O15Decay+F18Decay) << endl;   
  // cout << " There are " << ndecay << " recorded decays" << endl;
   cout << " There are " << ntrue << " true unscattered coincidences" << endl;
   cout << " There are " << nrandom << " random coincidences" << endl;
   cout << " There are " << nscatter << " scattered coincidences" << endl;
   cout << "  ==> there are " << nentries << " coincidences (true, scattered, and random)" << endl;   
   cout << "  ==> global scatter fraction = " << (float)nscatter/(float)(nentries-nrandom) << endl;
  // cout << "  ==> absolute sensitivity = " << 100.*(float)ntrue/ndecay << " % " << endl;
   Double_t p1 = e1->GetParameter(1);
   cout << " Measured O-15 life-time = " << -log(2.)/p1 << " [s]" << endl;
   cout << " Nominal  O-15 life-time = " << O15HalfLife <<"  [s]" << endl;
   cout << "   ==> difference = " << 100.*((-log(2.)/p1) - O15HalfLife)/O15HalfLife << " %" << endl;
   //Double_t p2 = g1->GetParameter(2);
   //cout << " Gamma acolinearity FWHM = " << p2*2.3548 << " degree (expected: 0.58)" << endl;
   c1 = new TCanvas("c1","GATE",3,28,970,632);
   Int_t pos=1;
   c1->Divide(3,2);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->cd(pos++);
   DetectPos1->Draw();
   //c1->cd(pos++);
   DetectPos2->SetMarkerColor(kRed);
   DetectPos2->Draw("same");
   tex = new TLatex(10.,200.,"Run 1");
   tex->SetTextColor(1);
   tex->SetLineWidth(2);
   tex->Draw();
   tex1 = new TLatex(10.,100.,"Run 2");
   tex1->SetTextColor(2);
   tex1->SetLineWidth(2);
   tex1->Draw();
   c1->cd(pos++);
   DetectPosAxial->Draw();
   c1->cd(pos++);
   DecayO15->SetLineColor(kRed);
   DecayO15->Draw();
   DecayF18->SetLineColor(kBlue);
   DecayF18->Draw("SAME");
   c1->cd(pos++);
   AxialSensitivityDet->Draw();
   c1->cd(pos++);
   ScatterFractionAxialDet->Divide(AxialScattersDet,AxialProfileDet,1.,1.,"");  
   ScatterFractionAxialDet->Draw();
   //c1->cd(pos++);
   //Acolinea_Angle_Distribution_deg->Draw();
   c1->cd(pos++);
   delay->Draw("time1");
   tex = new TLatex(30.,440.,"Delays: with coincidence sorter");
   tex->SetTextColor(1);
   tex->SetLineWidth(2);
   tex->Draw();
   tex1 = new TLatex(30.,410.,"Randoms: eventID1 != eventID2");
   tex1->SetTextColor(2);
   tex1->SetLineWidth(2);
   tex1->Draw();
   Coincidences->SetLineColor(2);
   Coincidences->Draw("time1","eventID1 != eventID2","same"); 

   
   c1->Update();   
   c1->SaveAs("benchmarkPET.gif");
}	
