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

// open the output file 
TFile* f =new TFile (" UserSpectrumExample.root"); 
// and print the content
f.ls();

// retrieve the energy spectrum histogram 
TH1F * h1 = (TH1F*)f.Get("energySpectrum");
h1->SetLineColor(kBlue);
// set title and the axis legend
h1->SetTitle("Different modes of User Spectrum source");
h1->GetXaxis()->SetTitle("Energy (MeV)");
h1->GetYaxis()->SetTitle("Counts");
h1->GetXaxis()->SetRangeUser(0. ,6);

// statistic type ( (1) write mean value, number of entries and RMS)
h1->SetStats(1);

// draw histogram using point (P) associated with error bar (E) or HIST for histogramm
h1->Draw("HIST"); 

}