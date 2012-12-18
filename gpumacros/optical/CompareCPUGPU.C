
{
  gROOT->Reset();

  // open output files:
   TFile* filecpu         = new TFile("./cpu_1M.root");
   TFile* filegpu         = new TFile("./gpu_1M.root");

  Float_t X, Y, Z;
  Float_t dX, dY, dZ;

  // create histograms
  filecpu->cd();
  TTree* tree_cpu = (TTree*)gDirectory->Get("PhaseSpace");

  tree_cpu->SetBranchAddress("X", &X);
  tree_cpu->SetBranchAddress("Y", &Y);
  tree_cpu->SetBranchAddress("Z", &Z);
  tree_cpu->SetBranchAddress("dX", &dX);
  tree_cpu->SetBranchAddress("dY", &dY);
  tree_cpu->SetBranchAddress("dZ", &dZ);

  TH1F* X_cpu  = new TH1F("X_cpu", "", 100, -5, 5);
  TH1F* Y_cpu  = new TH1F("Y_cpu", "", 100, -5, 5);
  TH1F* Z_cpu  = new TH1F("Z_cpu", "", 100, -5, 5);
  TH1F* dX_cpu  = new TH1F("dX_cpu", "", 50, -1.2, 1.2);
  TH1F* dY_cpu  = new TH1F("dY_cpu", "", 50, -1.2, 1.2);
  TH1F* dZ_cpu  = new TH1F("dZ_cpu", "", 50, -1.2, 1.2);

  Int_t num_cpu  = tree_cpu->GetEntries();

  for (Int_t i = 0; i < num_cpu; i++)
  {
    tree_cpu->GetEntry(i);
    X_cpu->Fill(X);
    Y_cpu->Fill(Y);
    Z_cpu->Fill(Z);
    dX_cpu->Fill(dX);
    dY_cpu->Fill(dY);
    dZ_cpu->Fill(dZ);
  }

  filegpu->cd();
  TTree* tree_gpu = (TTree*)gDirectory->Get("PhaseSpace");

  tree_gpu->SetBranchAddress("X", &X);
  tree_gpu->SetBranchAddress("Y", &Y);
  tree_gpu->SetBranchAddress("Z", &Z);
  tree_gpu->SetBranchAddress("dX", &dX);
  tree_gpu->SetBranchAddress("dY", &dY);
  tree_gpu->SetBranchAddress("dZ", &dZ);

  TH1F* X_gpu  = new TH1F("X_gpu", "", 100, -5, 5);
  TH1F* Y_gpu  = new TH1F("Y_gpu", "", 100, -5, 5);
  TH1F* Z_gpu  = new TH1F("Z_gpu", "", 100, -5, 5);
  TH1F* dX_gpu  = new TH1F("dX_gpu", "", 50, -1.2, 1.2);
  TH1F* dY_gpu  = new TH1F("dY_gpu", "", 50, -1.2, 1.2);
  TH1F* dZ_gpu  = new TH1F("dZ_gpu", "", 50, -1.2, 1.2);

  Int_t num_gpu  = tree_gpu->GetEntries();

  for (Int_t i = 0; i < num_gpu; i++)
  {
    tree_gpu->GetEntry(i);
    X_gpu->Fill(X);
    Y_gpu->Fill(Y);
    Z_gpu->Fill(Z);
    dX_gpu->Fill(dX);
    dY_gpu->Fill(dY);
    dZ_gpu->Fill(dZ);
  }


// Drawing 

//  gStyle->SetOptStat("e");
  gStyle->SetOptStat(0);
  gStyle->SetTitleFontSize(0.0);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetStatBorderSize(0);
  gROOT->ForceStyle();

// DRAW POSITIONS:

  TCanvas* canvas2 = new TCanvas();
  canvas2->SetFillColor(kWhite);
  canvas2->SetBorderMode(0);
  canvas2->Divide(3,1);

  canvas2->cd(1);
  X_cpu->Draw("p");
  X_cpu->SetMarkerColor(4);
  X_cpu->SetMarkerStyle(21);
  X_cpu->SetMarkerSize(0.5);
  X_gpu->Draw("psame");
  X_gpu->SetMarkerColor(8);
  X_gpu->SetMarkerStyle(21);
  X_gpu->SetMarkerSize(0.5);

  canvas2->cd(2);
  Y_cpu->Draw("p");
  Y_cpu->SetMarkerColor(4);
  Y_cpu->SetMarkerStyle(21);
  Y_cpu->SetMarkerSize(0.5);
  Y_gpu->Draw("psame");
  Y_gpu->SetMarkerColor(8);
  Y_gpu->SetMarkerStyle(21);
  Y_gpu->SetMarkerSize(0.5);

  canvas2->cd(3);
  Z_cpu->Draw("p");
  Z_cpu->SetMarkerColor(4);
  Z_cpu->SetMarkerStyle(21);
  Z_cpu->SetMarkerSize(0.5);
  Z_gpu->Draw("psame");
  Z_gpu->SetMarkerColor(8);
  Z_gpu->SetMarkerStyle(21);
  Z_gpu->SetMarkerSize(0.5);

//creating a legend
  TLegend *legend = new TLegend(0.6,0.4,0.8,0.6);
  legend->SetFillColor(kWhite);
  legend->AddEntry(Z_cpu,"cpu", "p");
  legend->AddEntry(Z_gpu,"gpu", "p");
  legend->Draw();

  canvas2->cd(0);

// DRAW DIRECTIONS:

  TCanvas* canvas3 = new TCanvas();
  canvas3->SetFillColor(kWhite);
  canvas3->Divide(3,1);
  canvas3->SetBorderMode(0);

  canvas3->cd(1);
  dX_cpu->Draw("p");
  dX_cpu->SetMarkerColor(4);
  dX_cpu->SetMarkerStyle(21);
  dX_cpu->SetMarkerSize(0.5);
  dX_gpu->Draw("psame");
  dX_gpu->SetMarkerColor(8);
  dX_gpu->SetMarkerStyle(21);
  dX_gpu->SetMarkerSize(0.5);

  canvas3->cd(2);
  dY_cpu->Draw("p");
  dY_cpu->SetMarkerColor(4);
  dY_cpu->SetMarkerStyle(21);
  dY_cpu->SetMarkerSize(0.5);
  dY_gpu->Draw("psame");
  dY_gpu->SetMarkerColor(8);
  dY_gpu->SetMarkerStyle(21);
  dY_gpu->SetMarkerSize(0.5);

  canvas3->cd(3);
  dZ_cpu->Draw("p");
  dZ_cpu->SetMarkerColor(4);
  dZ_cpu->SetMarkerStyle(21);
  dZ_cpu->SetMarkerSize(0.5);
  dZ_gpu->Draw("psame");
  dZ_gpu->SetMarkerColor(8);
  dZ_gpu->SetMarkerStyle(21);
  dZ_gpu->SetMarkerSize(0.5);

//creating a legend
  TLegend *legend2 = new TLegend(0.6,0.4,0.8,0.6);
  legend2->SetFillColor(kWhite);
  legend2->AddEntry(dZ_cpu,"cpu", "p");
  legend2->AddEntry(dZ_gpu,"gpu", "p");
  legend2->Draw();

  canvas3->cd(0);

 

/*
  pt = new TPaveText(0.6,0.83,0.87,0.88, "NDC"); // NDC sets coords
                                              // relative to pad dimensions
//   pt = new TPaveText(0.2,0.83,0.47,0.88, "NDC");   // for Absorption

  pt->SetFillColor(0); // text is black on white
  pt->SetTextSize(0.04); 
  pt->SetTextAlign(12);

  //text = pt->AddText("#gamma polarisation (1,0,0)");
  //text = pt->AddText("#gamma polarisation (0,1,0)");
  text = pt->AddText("#gamma polarisation (0,0,1)");

  pt->Draw();       //to draw your text object
*/

//  canvas->SaveAs("Rayleigh_vs_PhotonEnergy_PolarisationZ.gif");
//  canvas->SaveAs("Absorption_vs_PhotonEnergy_PolarisationX.gif");  // for Absorption

//  hCrystalXY->SetTitle("Crystal Hits");
//  canvasXY->SaveAs("./XYProjection.gif");


}
