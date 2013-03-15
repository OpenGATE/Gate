#include<iostream.h>
#include "TStyle.h"
#include "TChain.h"
#include "TLegend.h"
#include "TLatex.h"

void MakeFluorescencePlots(){

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
  gStyle->SetTitleSize(0.04,"x");
  gStyle->SetTitleSize(0.04,"y");
  gStyle->SetTitleFont(42,"x");
  gStyle->SetTitleFont(42,"y");

  ////////////// DECLARES HISTOGRAMS /////////////////////////////////////////////////////////////
  TH2D * histXY_EX = new TH2D("histXY_EX","",200,-15,15,200,-15,15);
  TH2D * histXY_EM = new TH2D("histXY_EM","",200,-15,15,200,-15,15);

  ///////////////////// DECLARES CHAINS ////////////////////////////////////////////////////////////
  TChain *chain1 = new TChain("OpticalData","");  

  ///////////////////////////// ADDS NTUPLES TO CHAINS ////////////////////////////////////////////

  chain1->Add("./Fluorescence_skin.root/OpticalData");

  //////////////////////////////////// MAKES SELECTIONS IN EACH CHAIN //////////////////////////////////////////////////////////// 
  chain1->Draw("CrystalLastHitPos_Y:CrystalLastHitPos_X>>histXY_EX","CrystalLastHitPos_Z == 30.0 && CrystalLastHitEnergy > 0.0","goff"); 
  chain1->Draw("CrystalLastHitPos_Y:CrystalLastHitPos_X>>histXY_EM","CrystalLastHitPos_Z == 30.0 && CrystalLastHitEnergy > 0.0 && CrystalLastHitEnergy < 0.00000219 ","goff"); 

   TCanvas *XY_EX = new TCanvas("XY_EX", "2D image of the excitation light",0,22,800,802);
   XY_EX->Range(-125,-125,125,125);
   XY_EX->SetGridx();
   XY_EX->SetGridy();
   XY_EX->SetTickx(1);
   XY_EX->SetTicky(1);
   XY_EX->SetFillColor(kWhite);
   TPaletteAxis *palette_EX = new TPaletteAxis(13.23807,-14.97093,14.93405,15.01938,histXY_EX);
   palette_EX->SetLabelColor(1);
   palette_EX->SetLabelOffset(0.005);
   palette_EX->SetLabelSize(0.03);
   palette_EX->SetTitleOffset(1);
   palette_EX->SetFillColor(0);
   histXY_EX->GetListOfFunctions()->Add(palette_EX,"br");
   histXY_EX->GetXaxis()->SetTitle("x-position(mm)");
   histXY_EX->GetYaxis()->SetTitle("y-position(mm)");
   histXY_EX->Draw("colz");

   TCanvas *XY_EM = new TCanvas("XY_EM", "2D image of the fluorescent light",0,22,800,802);
   XY_EM->Range(-125,-125,125,125);
   XY_EM->SetGridx();
   XY_EM->SetGridy();
   XY_EM->SetTickx(1);
   XY_EM->SetTicky(1);
   XY_EM->SetFillColor(kWhite);
   TPaletteAxis *palette_EM = new TPaletteAxis(13.23807,-14.97093,14.93405,15.01938,histXY_EM);
   palette_EM->SetLabelColor(1);
   palette_EM->SetLabelOffset(0.005);
   palette_EM->SetLabelSize(0.03);
   palette_EM->SetTitleOffset(1);
   palette_EM->SetFillColor(0);
   histXY_EM->GetListOfFunctions()->Add(palette_EM,"br");
   histXY_EM->GetXaxis()->SetTitle("x-position(mm)");
   histXY_EM->GetYaxis()->SetTitle("y-position(mm)");
   histXY_EM->Draw("colz");


}

