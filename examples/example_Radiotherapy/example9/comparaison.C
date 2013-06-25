{
  TFile * fileEGS = new TFile("data/EGS.root","OPEN");
  TFile * fileDose = new TFile("output/Electron-20MeV-Dose.root","OPEN");
  TFile * fileDoseInc = new TFile("output/Electron-20MeV-Dose-Uncertainty.root","OPEN");

  std::ifstream file("output/counterElectron-20MeV.txt");
  if (file.fail()){//file
    cout << "Probleme d'ouverture"<<endl;
    continue;
  }
  double n;
  TString texte;
  file>>texte;
  file>>texte>>texte>>texte>>texte;
  file>>texte>>texte>>texte>>texte;
  file>>texte>>texte>>texte>>texte;
  file>>texte>>texte>>texte>>texte;
  file>>texte;
  file>>n;

  file.close();
  n /= (1.5*1.5);

  TH1D * hEGS;
  fileEGS->GetObject("histo",hEGS) ;
  cout<<hEGS->GetNbinsX()<<endl;
  TH1F * hDose;
  fileDose->GetObject("histo",hDose) ;
  TH1F * hDoseInc;
  fileDoseInc->GetObject("histo",hDoseInc) ;

  TH1D * hGATE = new TH1D("gate","Dose", hEGS->GetNbinsX(), hEGS->GetBinLowEdge(1), hEGS->GetBinLowEdge(hEGS->GetNbinsX()) + hEGS->GetBinWidth(1)  );
  TH1D * hComp = new TH1D("comp","Comp", hEGS->GetNbinsX(), hEGS->GetBinLowEdge(1), hEGS->GetBinLowEdge(hEGS->GetNbinsX()) + hEGS->GetBinWidth(1)  );
  TH1D * hComp2 = new TH1D("comp2","Comp2", hEGS->GetNbinsX(), hEGS->GetBinLowEdge(1), hEGS->GetBinLowEdge(hEGS->GetNbinsX()) + hEGS->GetBinWidth(1)  );

  
  for(int i = 1 ; i < hDose->GetNbinsX()+1 ; i++)
  {
    hGATE->SetBinContent(i,hDose->GetBinContent(hDose->GetNbinsX()-i+1)/n);
    hGATE->SetBinError(i,hDoseInc->GetBinContent(hDose->GetNbinsX()-i+1)*hGATE->GetBinContent(hDose->GetNbinsX()-i+1) );
  }
  hGATE->Sumw2();
  hEGS->Sumw2();
  hComp->Sumw2();
  hComp2->Sumw2();

  hGATE->SetMarkerStyle(8);
  hGATE->SetMarkerSize(0.5);
  hGATE->SetMarkerColor(kRed);
  hGATE->SetLineColor(kRed);
  hGATE->SetLineWidth(2);
  hEGS->SetLineWidth(2);
  hGATE->SetFillColor(19);
  hEGS->SetFillColor(19);

  hComp->Add(hEGS, hGATE, 1, -1);
  hComp->Divide(hComp, hEGS,100,1);

 hComp2->Add(hEGS, hGATE, 1, -1);
  hComp2->Scale( 1./hEGS->GetMaximum()*100);

  hComp2->GetXaxis()->SetTitle("Depth (cm)");
  hComp2->GetYaxis()->SetTitle("#frac{Dose_{EGS} - Dose_{GATE}}{Dose_{Max}}    (%)");



  hGATE->GetXaxis()->SetTitle("Depth (cm)");
  hGATE->GetYaxis()->SetTitle("Dose/fluence   (Gy #upoint cm^{2})");
  //hGATE->GetYaxis()->SetRangeUser(0, 11e-12);
  hGATE->GetXaxis()->SetRangeUser(0, 15);

  hComp->GetXaxis()->SetTitle("Depth (cm)");
  hComp->GetYaxis()->SetTitle("#frac{Dose_{EGS} - Dose_{GATE}}{Dose_{EGS}}    (%)");
  hComp->GetXaxis()->SetRangeUser(0, 15);
  hComp->GetYaxis()->SetRangeUser(-20, 14);
  hComp2->GetXaxis()->SetTitle("Depth (cm)");
  hComp2->GetYaxis()->SetTitle("#frac{Dose_{EGS} - Dose_{GATE}}{Dose_{Max}}    (%)");
  hComp2->GetXaxis()->SetRangeUser(0, 15);
  hComp2->GetYaxis()->SetRangeUser(-12, 5);

  TCanvas * c1 = new TCanvas("c1","Comparison of GATE and EGS results");
  c1->Divide(1,3);
  c1->cd(1);
  hGATE->Draw("EBAR");
  hEGS->Draw("E same");

  TLine *line1 = new TLine(2, 0, 2, 3.38e-10);
  line1->SetLineColor(kBlue);
  line1->SetLineStyle(2);
  line1->Draw();
  TLine *line2 = new TLine(3, 0, 3, 3.38e-10);
  line2->SetLineColor(kBlue);
  line2->SetLineStyle(2);
  line2->Draw();
  TLine *line3 = new TLine(6, 0, 6, 3.38e-10);
  line3->SetLineColor(kBlue);
  line3->SetLineStyle(2);
  line3->Draw();



  TLegend * leg = new TLegend(0.6,0.6,0.8,0.8);
  leg->AddEntry(hGATE,"GATE","lp");
  leg->AddEntry(hEGS,"EGS","lp");
  leg->SetFillColor(0);
  leg->Draw();

  c1->cd(2);
  hComp->Draw();
  TLine *line1b = new TLine(2, -20., 2,14);
  line1b->SetLineColor(kBlue);
  line1b->SetLineStyle(2);
  line1b->Draw();
  TLine *line2b = new TLine(3, -20., 3, 14 );
  line2b->SetLineColor(kBlue);
  line2b->SetLineStyle(2);
  line2b->Draw();
  TLine *line3b = new TLine(6, -20., 6, 14 );
  line3b->SetLineColor(kBlue);
  line3b->SetLineStyle(2);
  line3b->Draw();

 c1->cd(3);
  hComp2->Draw();
  TLine *line1c = new TLine(2, -12., 2,5);
  line1c->SetLineColor(kBlue);
  line1c->SetLineStyle(2);
  line1c->Draw();
  TLine *line2c = new TLine(3, -12., 3, 5 );
  line2c->SetLineColor(kBlue);
  line2c->SetLineStyle(2);
  line2c->Draw();
  TLine *line3c = new TLine(6, -12., 6, 5 );
  line3c->SetLineColor(kBlue);
  line3c->SetLineStyle(2);
  line3c->Draw();

}
