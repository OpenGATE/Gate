//#include<iostream.h>
#include "TStyle.h"
#include "TChain.h"
#include "TLegend.h"
#include "TLatex.h"

void benchAnalyseOptical(){

  gStyle->SetOptStat(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetLabelSize(0.03,"x");
  gStyle->SetLabelSize(0.03,"y");
  gStyle->SetLabelFont(42,"x");
  gStyle->SetLabelFont(42,"y");
  gStyle->SetOptStat(1111);


  ////////////// DECLARES HISTOGRAMS /////////////////////////////////////////////////////////////
  TH1D * X = new TH1D("X","",200,-40,40);
  TH1D * Y = new TH1D("Y","",200,-40,40);
  TH1D * Z = new TH1D("Z","",200,-40,40);
  TH1D * Energy = new TH1D("Energy","",50,0,0.000004);
  TH2D * histXY = new TH2D("histXY","",200,-40,40,200,-40,40);

  ///////////////////// DECLARES CHAINS ////////////////////////////////////////////////////////////
  TChain *chain_fluo = new TChain("OpticalData",""); 

  ///////////////////////////// ADDS NTUPLES TO CHAINS ////////////////////////////////////////////
  chain_fluo->Add("optical_results.root/OpticalData");

  //////////////////////////////////// MAKES SELECTIONS IN EACH CHAIN ////////////////////////////////////////////////////////////
  chain_fluo->Draw("CrystalLastHitPos_X>>X","","goff"); 
  chain_fluo->Draw("CrystalLastHitPos_Y>>Y","","goff"); 
  chain_fluo->Draw("CrystalLastHitPos_Z>>Z","","goff"); 
  chain_fluo->Draw("CrystalLastHitEnergy>>Energy","","goff");
  chain_fluo->Draw("CrystalLastHitPos_Y:CrystalLastHitPos_X>>histXY","","goff");

  //////////////////////////////////// IF YOU WANT TO USE THE HITS TREE INFORMATION //////////////////////////////////////////////
/*
  TChain *chain_fluo = new TChain("Hits",""); 
  chain_fluo->Add("/home/vc232495/WORKDIR/Gatev6.2/opengate/examples/example_OPTICAL/sbvlExperienceBiolumi/Bioluminescence.root/Hits");
  chain_fluo->Draw("posX>>X","","goff"); 
  chain_fluo->Draw("posY>>Y","","goff"); 
  chain_fluo->Draw("posZ>>Z","","goff"); 
  chain_fluo->Draw("edep>>Energy","","goff");
  chain_fluo->Draw("posY:posX>>histXY","","goff");
*/

  // Drawing 

  TCanvas* c_energy = new TCanvas("c_energy"," ",0,0,500,500);
  c_energy->SetFillColor(kWhite);
  Energy->Draw();
  Energy->SetLineColor(28);
  Energy->SetFillColor(28);
  Energy->GetXaxis()->SetTitle("Optical photon Energy (MeV)");
//  c_energy->SaveAs("energy.gif");

  TCanvas* c_positions = new TCanvas("c_positions"," ",0,0,1200,500);
  c_positions->SetFillColor(kWhite);
  c_positions->Divide(3,1);
  c_positions->cd(1);
  X->Draw();
  X->GetXaxis()->SetTitle("Optical photon X-position");
  X->SetLineColor(46);
  X->SetFillColor(46);
  c_positions->cd(2);
  Y->Draw();
  Y->GetXaxis()->SetTitle("Optical photon Y-position");
  Y->SetLineColor(36);
  Y->SetFillColor(36);
  c_positions->cd(3);
  Z->Draw();
  Z->SetLineColor(30);
  Z->SetFillColor(30);
  Z->GetXaxis()->SetTitle("Optical photon Z-position");
  c_positions->cd(0);
//  c_positions->SaveAs("positionsXYZ.gif");

   TCanvas *XY = new TCanvas("XY", " ",0,22,800,802);
   XY->Range(-125,-125,125,125);
   XY->SetGridx();
   XY->SetGridy();
   XY->SetTickx(1);
   XY->SetTicky(1);
   XY->SetFrameBorderMode(0);
   XY->SetFrameBorderMode(0);
   XY->SetFillColor(kWhite);
 
   TPaletteAxis *palette = new TPaletteAxis(88.88191,-100.0648,100.1884,99.74093,histXY);
   palette->SetLabelColor(1);
   palette->SetLabelFont(42);
   palette->SetLabelOffset(0.005);
   palette->SetLabelSize(0.03);
   palette->SetTitleOffset(1);
   palette->SetTitleSize(0.02);
   palette->SetFillColor(1);
   palette->SetFillStyle(1001);
//   histXY->GetListOfFunctions()->Add(palette,"br");
   
   TPaveStats *ptstats = new TPaveStats(0.1281407,0.6498708,0.2826633,0.869509,"brNDC");
   ptstats->SetName("stats");
   ptstats->SetBorderSize(1);
   ptstats->SetFillColor(0);
   ptstats->SetTextAlign(12);
   ptstats->SetTextFont(42);
   ptstats->SetOptStat(1111);
   ptstats->SetOptFit(0);
   ptstats->Draw();

   histXY->GetListOfFunctions()->Add(ptstats);
   ptstats->SetParent(histXY->GetListOfFunctions());
   histXY->GetXaxis()->SetTitle("Optical photon X-position (mm)");
   histXY->GetXaxis()->SetLabelFont(42);
   histXY->GetXaxis()->SetTitleFont(42);
   histXY->GetYaxis()->SetTitle("Optical photon Y-position (mm)");
   histXY->GetYaxis()->SetLabelFont(42);
   histXY->GetYaxis()->SetTitleOffset(1.43);
   histXY->GetYaxis()->SetTitleFont(42);
   histXY->Draw("colz");
   histXY->GetXaxis()->SetTitleSize(0.03);
   histXY->GetYaxis()->SetTitleSize(0.03);  
   histXY->GetXaxis()->SetLabelSize(0.03);
   histXY->GetYaxis()->SetLabelSize(0.03);
   XY->Modified();
   XY->cd();
   XY->SetSelected(XY);

//   XY->SaveAs("projectionXY.gif");


}
