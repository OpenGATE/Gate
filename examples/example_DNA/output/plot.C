// *********************************************************************
// To execute this macro under ROOT, 
//   1 - launch ROOT (usually type 'root' at your machine's prompt)
//   2 - type '.X plot.C' at the ROOT session prompt
//   by Q.T.PHAM and Y.PERROT
// *********************************************************************
{
gROOT->Reset();
gStyle->SetPalette(1);
gROOT->SetStyle("Plain");
Double_t scale;
	
c1 = new TCanvas ("c1","graph",66,52,1000,500);


TFile* f =new TFile ("Target1.root"); 
f.ls();
 
TH1F * h1 = (TH1F*)f.Get("energySpectrum");
h1->SetLineColor(kBlue);
h1->SetTitle("Production of Secondary Particles");
h1->GetXaxis()->SetTitle("Edep (MeV)");
h1->GetYaxis()->SetTitle("Frequency");
h1->GetXaxis()->SetRangeUser(0.001,0.007);
h1->SetStats(0);
h1->Draw("PE"); 
  
TFile* f2 =new TFile ("Target2.root");
TH1F * h2 = (TH1F*)f2.Get("energySpectrum");
h2->SetLineColor(kRed);
h2->Draw("PESAME"); 

TFile* f3 =new TFile ("Target3.root");
TH1F * h3 = (TH1F*)f3.Get("energySpectrum");
h3->SetLineColor(kGreen);
h3->Draw("PESAME"); 

leg = new TLegend(0.9,0.7,0.48,0.9);
leg->SetFillColor(kWhite);
leg->AddEntry(h1,"Target1","l");
leg->AddEntry(h2,"Target2","l");
leg->AddEntry(h3,"Target3","l");
leg->Draw();

}


