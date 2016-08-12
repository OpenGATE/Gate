// *********************************************************************
// To execute this macro under ROOT, 
//   1 - launch ROOT (usually type 'root' at your machine's prompt)
//     - type '.X plot.C' at the ROOT session prompt
//	 2 - in your terminal type 'root plot.C'
//   by Y. PERROT, S. NICOLAS 
//   
// *********************************************************************

{
// root initialisation for graphical output
gROOT->Reset(); 
gStyle->SetPalette(1);
gROOT->SetStyle("Plain");

// create a new canvas
c1 = new TCanvas ("c1","graph",66,52,1000,500);
c1->Divide(2,1);

// open the output file 
TFile* f =new TFile (" fastY90Brem.root"); 
// and print the content
f->ls();

// retrieve the energy spectrum histogram 
TH1F * h1 = (TH1F*)f->Get("energySpectrum");
h1->SetLineColor(kBlue);
// set title and the axis legend
h1->SetTitle("fastY90 bremsstrahlung spectrum");
h1->GetXaxis()->SetTitle("Energy (MeV)");
h1->GetYaxis()->SetTitle("Counts");
h1->GetXaxis()->SetRangeUser(0. ,1);

// statistic type ( (1) write mean value, number of entries and RMS)
h1->SetStats(1);

// draw histogram using point (P) associated with error bar (E) or HIST for histogramm
c1->cd(1);
h1->Draw("HIST"); 

// open the output file 
TFile* f2 =new TFile (" fastY90Pos.root"); 
// and print the content
f2->ls();

// retrieve the energy spectrum histogram 
TH1F * h2 = (TH1F*)f2->Get("energySpectrum");
h2->SetLineColor(kRed);
// set title and the axis legend
h2->SetTitle("fastY90 positron kinetic energy");
h2->GetXaxis()->SetTitle("Energy (MeV)");
h2->GetYaxis()->SetTitle("Counts");
h2->GetXaxis()->SetRangeUser(0. ,1);

// statistic type ( (1) write mean value, number of entries and RMS)
h2->SetStats(1);

// draw histogram using point (P) associated with error bar (E) or HIST for histogramm
c1->cd(2);
h2->Draw("HIST"); 

}
