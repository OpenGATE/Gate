
{
  gROOT->Reset();
  // open the files
  TFile* file         = new TFile("gate.root");
  TFile* reffile      = new TFile("optical1_reference.root");
  // read in the reference histograms
  TH1F*  refspectrum  = (TH1F*)reffile.Get("spectrum");
  TH2I*  refcrosstalk = (TH2I*)reffile.Get("crosstalk");
  // create the histograms from the simulation data
  file->cd();
  TTree* singlestree = (TTree*)gDirectory->Get("Singles");
  Float_t energy;
  Int_t   pixelid;
  singlestree->SetBranchAddress("energy", &energy);
  singlestree->SetBranchAddress("level2ID", &pixelid);
  TH1F* spectrum  = new TH1F("spectrum", "Energy spectrum", 100, 0, 11000);
  TH2I* crosstalk = new TH2I("crosstalk", "Crosstalk/inter-crystal scatter", 10, -0.5, 9.5, 10, -0.5, 9.5);
  Int_t nsingles  = singlestree->GetEntries();
  for (Int_t i = 0; i < nsingles; i++)
  {
    singlestree->GetEntry(i);
    Int_t y = pixelid/9;
    Int_t x = pixelid - y*9;
    spectrum->Fill(energy);
    crosstalk->Fill(x, y);
  }
  // compare both spectra
  cout << endl;
  cout << "==========================================================" << endl;
  cout << "=== GATE optical 1 results                             ===" << endl;
  cout << "==========================================================" << endl << endl;

  cout << "Comparing energy spectrum to reference energy spectrum" << endl;
  cout << "==========================================================" << endl;
  Double_t result = spectrum->Chi2Test(refspectrum, "P");
  cout << "probability value = " << result << endl << endl;
  if (result > 0.05) cout << "The spectra are equal. Good!" << endl;
  else cout << "The spectra differ. Not good!" << endl;
  cout << endl << endl;

  cout << "Comparing the crosstalk/inter-crystal scatter to reference" << endl;
  cout << "==========================================================" << endl;
  Double_t result = crosstalk->Chi2Test(refcrosstalk, "P");
  cout << "probability value = " << result << endl << endl;
  if (result > 0.05) cout << "The spectra are equal. Good!" << endl;
  else cout << "The spectra differ. Not good!" << endl;
  cout << endl << endl;

  // plot both spectra

  TCanvas* canvas = new TCanvas();
  canvas->Divide(2,2);
  canvas->cd(1);
  Double_t scalef = 1/refspectrum->Integral();
  refspectrum->Scale(scalef);
  scalef = 1/spectrum->Integral();
  spectrum->Scale(scalef);
  spectrum->SetLineColor(kRed);
  refspectrum->GetXaxis()->SetTitle("energy (number of photons detected)");
  refspectrum->GetYaxis()->SetTitle("normalised counts/channel");
  refspectrum->Draw();
  spectrum->Draw("same");
  canvas->cd(2);
  scalef = crosstalk->Integral()/refcrosstalk->Integral();
  refcrosstalk->Scale(scalef);
  refcrosstalk->SetMarkerSize(1.6);
  refcrosstalk->SetTitle("Crosstalk/inter-crystal scatter, normalised reference");
  refcrosstalk->Draw("text");
  canvas->cd(4);
  crosstalk->SetMarkerSize(1.6);
  crosstalk->Draw("text");
  canvas->Print("optical1.eps");
}
