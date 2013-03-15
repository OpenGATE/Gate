#include<iostream.h>
#include "TStyle.h"
#include "TChain.h"
#include "TLegend.h"
#include "TLatex.h"

void MakeBioluminescencePlots(){

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
  TH2D * histXY = new TH2D("histXY","",200,-15,15,200,-15,15);

  ///////////////////// DECLARES CHAINS ////////////////////////////////////////////////////////////
  TChain *chain1 = new TChain("OpticalData","");  

  ///////////////////////////// ADDS NTUPLES TO CHAINS ////////////////////////////////////////////

  chain1->Add("./Bioluminescence_skin.root/OpticalData");

  //////////////////////////////////// MAKES SELECTIONS IN EACH CHAIN //////////////////////////////////////////////////////////// 
  chain1->Draw("CrystalLastHitPos_Y:CrystalLastHitPos_X>>histXY","CrystalLastHitPos_Z == 30.0 && CrystalLastHitEnergy > 0.0","goff"); 

   TCanvas *XY = new TCanvas("XY", " ",0,22,800,802);
   XY->Range(-125,-125,125,125);
   XY->SetGridx();
   XY->SetGridy();
   XY->SetTickx(1);
   XY->SetTicky(1);
   XY->SetFrameBorderMode(0);
   XY->SetFrameBorderMode(0);
   XY->SetFillColor(kWhite);
 

   TPaletteAxis *palette = new TPaletteAxis(13.23807,-14.97093,14.93405,15.01938,histXY);
   palette->SetLabelColor(1);
   palette->SetLabelFont(62);
   palette->SetLabelOffset(0.005);
   palette->SetLabelSize(0.02);
   palette->SetTitleOffset(1);
   palette->SetTitleSize(0.04);
   palette->SetFillColor(0);
   palette->SetFillStyle(1001);
   histXY->GetListOfFunctions()->Add(palette,"br");

   histXY->GetXaxis()->SetTitle("x-position(mm)");
   histXY->GetXaxis()->SetLabelFont(42);
   histXY->GetXaxis()->SetTitleFont(42);
   histXY->GetYaxis()->SetTitle("y-position(mm)");
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



}

