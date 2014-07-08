//ooooooooooOOOOOOO00000000000OOOOOOOOOOOooooooooooooo//
//
//ooooooooooOOOOOOO00000000000OOOOOOOOOOOooooooooooooo//

{
  gROOT->Reset();

  int nCol=1;      // n.column
  int nRow=1;      // n.row
  int nZ=400;        // n.z
  int imBitNumber=32;


  int              nPoint_tot = nCol*nRow*nZ;
  long             lSize = nPoint_tot*(imBitNumber/8);

  Double_t x1[nZ], y1[nZ];
  Double_t x2[nZ], y2[nZ];
  Double_t x3[nZ], y3[nZ];


  //... define the buffer
  float       *buffer_p;
  float       *buffer_g;
  float       *buffer_c;

  buffer_p = (float*) malloc (lSize);
  buffer_g = (float*) malloc (lSize);
  buffer_c = (float*) malloc (lSize);

  //... open the image
  FILE* protonFile;
  FILE* gammaFile;
  FILE* carbonFile;

  protonFile = fopen ("output/output-proton-Edep.raw" , "rb" );
  gammaFile = fopen ("output/output-gamma-Edep.raw" , "rb" );
  carbonFile = fopen ("output/output-carbon-Edep.raw" , "rb" );

  //... read the image
  fread (buffer_p,1,lSize,protonFile);
  fread (buffer_g,1,lSize,gammaFile);
  fread (buffer_c,1,lSize,carbonFile);


  for(int i = 0 ; i<nZ ; i++){
    x1[i]  = i;
    y1[i] = buffer_p[i];

    x2[i]  = i;
    y2[i] = buffer_g[i];

    x3[i]  = i;
    y3[i] = buffer_c[i];
  }

  TGraph *gr1 = new TGraph(nZ,x1,y1);
  gr1->SetLineColor(1);
  gr1->SetLineWidth(3);
  gr1->SetTitle("BenchRT energy curves");
  gr1->GetHistogram()->SetXTitle("Depth - mm");
  gr1->GetHistogram()->SetYTitle("Energy - MeV");
  gr1->GetXaxis()->SetTitleOffset(1.1);
  gr1->GetYaxis()->SetTitleOffset(1.35);

  TGraph *gr2 = new TGraph(nZ,x2,y2);
  gr2->SetLineColor(2);
  gr2->SetLineWidth(3);

  TGraph *gr3 = new TGraph(nZ,x3,y3);
  gr3->SetLineColor(4);
  gr3->SetLineWidth(3);

  //save root file with graph
  TFile *rootFile = new TFile("graph_ouput.root","recreate");
  gr1->Write();
  gr2->Write();
  gr3->Write();
  rootFile->Close();

  TLegend *leg = new TLegend(0.5,0.5,0.8,0.65);
  leg->SetFillColor(0);
  leg->SetTextSize(0.03);
  leg->AddEntry(gr3,"Carbon beam: 2500 MeV","lp");
  leg->AddEntry(gr1,"Proton beam: 150 MeV ","lp");
  leg->AddEntry(gr2,"Photon beam: 18 MeV","lp");

  TCanvas *c1 = new TCanvas("c1","transparent pad",200,10,700,500);
  c1->SetGrid();

  c1->cd();
  gr3->Draw("apl");
  //scale graph for drawing
  for (int i = 0 ; i<nZ ; i++) gr2.GetY()[i] *= 0.5*(gr3->GetHistogram()->GetMaximum()/gr2->GetHistogram()->GetMaximum());
  gr2->Draw("spl");
  for (int i = 0 ; i<nZ ; i++) gr1.GetY()[i] *= (gr3->GetHistogram()->GetMaximum()/gr1->GetHistogram()->GetMaximum());
  gr1->Draw("spl");
  leg->Draw("same");

  c1->SaveAs("output/benchmarkRT.gif");

  // fclose(protonFile);
  // fclose(gammaFile);
  // fclose(protonFile);
}

//ooooooooooOOOOOOO00000000000OOOOOOOOOOOooooooooooooo//
